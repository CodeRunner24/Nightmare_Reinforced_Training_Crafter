"""Bounded upstream Crafter run with external GPU sampling; no algorithm changes."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--steps', type=int, default=1280)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--timeout', type=int, default=900)
args = parser.parse_args()
out = ROOT / 'logs' / args.name
out.mkdir(parents=True, exist_ok=False)
command = [sys.executable, '-u', 'dreamerv3/main.py', '--configs', 'crafter',
           '--logdir', str(out), '--jax.platform', 'cuda', '--jax.prealloc', 'False',
           '--run.steps', str(args.steps), '--batch_size', str(args.batch_size),
           '--run.log_every', '10']
print('COMMAND:', ' '.join(command), flush=True)
started = time.monotonic()
phase = {'name': 'startup', 'training_start_s': None}
samples = []
query = ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu',
         '--format=csv,noheader,nounits']

def read_gpu():
    values = subprocess.check_output(query, text=True, timeout=5).strip().splitlines()[0].split(',')
    return [float(v.strip()) for v in values]

baseline = [read_gpu() for _ in range(3)]
print('BASELINE GPU MiB:', [x[0] for x in baseline], flush=True)
proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1,
                        start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})

def output_reader():
    with (out / 'terminal.log').open('w', buffering=1) as handle:
        for line in proc.stdout:
            elapsed = time.monotonic() - started
            if 'Compiling train and report' in line:
                phase['name'] = 'compilation'
            if 'Start training loop' in line:
                phase['name'] = 'training'
                phase['training_start_s'] = elapsed
            handle.write(line)
            print(line, end='', flush=True)

reader = threading.Thread(target=output_reader, daemon=True)
reader.start()
timed_out = False
last_status = -30
with (out / 'gpu_samples.csv').open('w', newline='', buffering=1) as handle:
    writer = csv.writer(handle)
    writer.writerow(['elapsed_s', 'phase', 'device_used_mib', 'device_total_mib',
                     'gpu_util_pct', 'power_w', 'temperature_c'])
    while proc.poll() is None:
        elapsed = time.monotonic() - started
        try:
            row = [round(elapsed, 3), phase['name'], *read_gpu()]
            samples.append(row)
            writer.writerow(row)
        except Exception as exc:
            print('GPU SAMPLE ERROR:', str(exc), flush=True)
        if elapsed - last_status >= 30:
            last_status = elapsed
            metricfile = out / 'metrics.jsonl'
            metrics = []
            if metricfile.exists():
                for line in metricfile.read_text().splitlines():
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            latest = metrics[-1] if metrics else {}
            print(f"[PROFILE] elapsed={elapsed:.1f}s cap={args.timeout}s phase={phase['name']} "
                  f"step={latest.get('step', 'not logged')} "
                  f"train_fps={latest.get('fps/train', 'not logged')} "
                  f"gpu_mib={samples[-1][2] if samples else 'unknown'}", flush=True)
        if elapsed >= args.timeout:
            timed_out = True
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            break
        time.sleep(1)
code = proc.wait()
reader.join(timeout=15)
result = {'command': command, 'returncode': code, 'timed_out': timed_out,
          'jax_memory_fraction': os.environ.get('XLA_CLIENT_MEM_FRACTION', 'default'),
          'elapsed_s': time.monotonic() - started, **phase,
          'baseline_device_mib': [x[0] for x in baseline],
          'peak_device_mib': max((x[2] for x in samples), default=None),
          'gpu_sample_count': len(samples),
          'memory_scope': 'whole GPU including desktop; 1s sampling, not per-process allocation'}
(out / 'profile.json').write_text(json.dumps(result, indent=2) + '\n')
print('PROFILE RESULT:', json.dumps(result, indent=2), flush=True)
sys.exit(0 if code == 0 else 1)
