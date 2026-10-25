"""Bounded 5K Crafter test with live GPU, RAM, disk and progress logging."""
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import psutil
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get(
    'CRAFTER_MEMORY_OUT', ROOT / 'logs' / 'crafter_memory_5k_01'))
OUT.mkdir(parents=True, exist_ok=False)
TARGET = 5000
TIMEOUT = 3600
batch_size = os.environ.get('CRAFTER_BATCH_SIZE', '12')
env = {**os.environ, 'XLA_CLIENT_MEM_FRACTION': '0.90',
       'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'PYTHONUNBUFFERED': '1'}
command = [sys.executable, '-u', 'setup/run_with_final_checkpoint.py',
           '--configs', 'crafter', '--logdir', str(OUT), '--batch_size', batch_size,
           '--jax.platform', 'cuda', '--jax.prealloc', 'False',
           '--run.steps', str(TARGET), '--run.log_every', '20']
query = ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu',
         '--format=csv,noheader,nounits']


def gpu_sample():
    return [float(x.strip()) for x in subprocess.check_output(
        query, text=True, timeout=5).strip().splitlines()[0].split(',')]


def disk_bytes():
    return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())


baseline = {'gpu_mib': gpu_sample()[0],
            'wsl_available_mib': psutil.virtual_memory().available / 2**20,
            'disk_free_gib': shutil.disk_usage(OUT).free / 2**30}
print('COMMAND:', ' '.join(command), flush=True)
print('BASELINE:', json.dumps(baseline), flush=True)
started = time.monotonic()
proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1,
                        start_new_session=True)
child = psutil.Process(proc.pid)
state = {'phase': 'startup', 'training_start_s': None}


def reader():
    with (OUT / 'terminal.log').open('w', buffering=1) as handle:
        for line in proc.stdout:
            if 'Compiling train and report' in line:
                state['phase'] = 'compilation'
            if 'Start training loop' in line:
                state['phase'] = 'training'
                state['training_start_s'] = time.monotonic() - started
            if 'FINAL CHECKPOINT:' in line:
                state['phase'] = 'final_checkpoint'
            handle.write(line)
            tqdm.write(line.rstrip())


thread = threading.Thread(target=reader, daemon=True)
thread.start()
samples = []
stop_reason = None
last_status = -30
metrics_seen = 0
current_disk_bytes = 0
with (OUT / 'resources.csv').open('w', newline='', buffering=1) as handle, \
        (OUT / 'metric_times.jsonl').open('w', buffering=1) as timing:
    writer = csv.writer(handle)
    writer.writerow(['elapsed_s', 'phase', 'gpu_used_mib', 'gpu_total_mib',
        'gpu_util_pct', 'power_w', 'temperature_c', 'wsl_available_mib',
        'process_tree_rss_mib', 'wsl_swap_used_mib', 'disk_free_gib',
        'logdir_size_mib', 'last_logged_step'])
    progress = tqdm(total=TARGET, desc='Crafter 5K', unit='step', mininterval=30)
    while proc.poll() is None:
        elapsed = time.monotonic() - started
        metrics = []
        file = OUT / 'metrics.jsonl'
        if file.exists():
            for line in file.read_text().splitlines():
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        for row in metrics[metrics_seen:]:
            timing.write(json.dumps({'observed_elapsed_s': elapsed, **row}) + '\n')
        metrics_seen = len(metrics)
        latest = next((m for m in reversed(metrics) if 'fps/policy' in m), {})
        step = latest.get('step', 0)
        try:
            ram = psutil.virtual_memory().available / 2**20
            processes = [child, *child.children(recursive=True)]
            rss = sum(p.memory_info().rss for p in processes if p.is_running()) / 2**20
            disk_free = shutil.disk_usage(OUT).free / 2**30
            if elapsed - last_status >= 30:
                current_disk_bytes = disk_bytes()
            row = [round(elapsed, 3), state['phase'], *gpu_sample(), ram, rss,
                   psutil.swap_memory().used / 2**20, disk_free,
                   current_disk_bytes / 2**20, step]
            samples.append(row)
            writer.writerow(row)
            if step > progress.n:
                progress.update(step - progress.n)
            if elapsed - last_status >= 30:
                last_status = elapsed
                fps = latest.get('fps/policy', 0)
                eta = (TARGET - step) / fps if fps > 0 and 'train/opt/loss' in latest else None
                tqdm.write(f"[MEMORY] elapsed={elapsed/60:.1f}min phase={state['phase']} "
                    f"step={step}/{TARGET} env_fps={fps:.3f} "
                    f"eta_min={eta/60 if eta else 'warming up'} "
                    f"GPU={row[2]:.0f}MiB RSS={rss:.0f}MiB WSL_free={ram:.0f}MiB "
                    f"disk={current_disk_bytes/2**30:.2f}GiB")
            if ram < 2048:
                stop_reason = 'WSL available RAM below 2048 MiB'
            elif disk_free < 8:
                stop_reason = 'Disk free space below 8 GiB'
            elif elapsed > TIMEOUT:
                stop_reason = 'One-hour pilot limit'
            if stop_reason:
                tqdm.write('STOP: ' + stop_reason)
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(2)
    code = proc.wait()
    thread.join(timeout=20)
    final_file = OUT / 'final_state.json'
    final = json.loads(final_file.read_text()) if final_file.exists() else None
    if final and final['final_step'] > progress.n:
        progress.update(final['final_step'] - progress.n)
    progress.close()
result = {'command': command, 'returncode': code, 'stop_reason': stop_reason,
          'elapsed_s': time.monotonic() - started, **state,
          'baseline': baseline, 'target_steps': TARGET,
          'final_step': final['final_step'] if final else None,
          'peak_total_gpu_mib': max((r[2] for r in samples), default=None),
          'peak_process_tree_rss_mib': max((r[8] for r in samples), default=None),
          'minimum_wsl_available_mib': min((r[7] for r in samples), default=None),
          'peak_wsl_swap_mib': max((r[9] for r in samples), default=None),
          'logdir_bytes': disk_bytes(), 'sample_interval_s': 2,
          'memory_scope': 'GPU is whole-device; RSS includes child process tree; RAM is WSL MemAvailable'}
(OUT / 'profile.json').write_text(json.dumps(result, indent=2) + '\n')
print('MEMORY TEST RESULT:', json.dumps(result, indent=2), flush=True)
sys.exit(0 if code == 0 and final and final['final_step'] == TARGET else 1)
