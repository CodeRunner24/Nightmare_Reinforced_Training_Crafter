"""Summarize saved measurements only; never starts training."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import math
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()
root = args.directory
profile = json.loads((root / 'profile.json').read_text())
with (root / 'gpu_samples.csv').open() as handle:
    gpu = list(csv.DictReader(handle))
metrics = []
if (root / 'metrics.jsonl').exists():
    for line in (root / 'metrics.jsonl').read_text().splitlines():
        metrics.append(json.loads(line))
logs = [r for r in metrics if 'fps/policy' in r]
train = [r for r in logs if 'train/opt/loss' in r]
# Upstream automatically profiles optimizer updates 100--120. Exclude log
# windows touched by that instrumentation, including start/stop trace writes.
profiler_active = False
profiler_touched = False
profiler_steps = set()
for line in (root / 'terminal.log').read_text().splitlines():
    if 'Start JAX profiler:' in line:
        profiler_active = True
        profiler_touched = True
    if 'Stop JAX profiler' in line:
        profiler_active = False
        profiler_touched = True
    match = re.search(r'\[Agent Step ([\d_]+)\]', line)
    if match:
        if profiler_active or profiler_touched:
            profiler_steps.add(int(match.group(1).replace('_', '')))
        profiler_touched = False
# Discard first loss-bearing window, which can mix replay fill and training.
windows = []
for prev, curr in zip(logs, logs[1:]):
    if 'train/opt/loss' not in prev or 'train/opt/loss' not in curr:
        continue
    if curr['step'] in profiler_steps:
        continue
    steps = curr['step'] - prev['step']
    fps = curr['fps/policy']
    if steps > 0 and fps > 0:
        windows.append({'end_step': curr['step'], 'steps': steps,
                        'env_fps': fps, 'elapsed_s': steps / fps})
baseline = statistics.mean(profile['baseline_device_mib'])
peak = max(float(r['device_used_mib']) for r in gpu)
train_gpu = [r for r in gpu if r['phase'] == 'training']
summary = {
    'returncode': profile['returncode'], 'timed_out': profile['timed_out'],
    'wall_seconds': profile['elapsed_s'],
    'startup_to_training_seconds': profile['training_start_s'],
    'last_logged_step': metrics[-1]['step'] if metrics else None,
    'last_logged_optimizer_updates': train[-1]['train/opt/updates'] if train else None,
    'parameter_count': train[-1].get('train/opt/param_count') if train else None,
    'baseline_device_mib': baseline, 'peak_device_mib': peak,
    'peak_above_baseline_mib': peak - baseline,
    'peak_during_training_device_mib': max((float(r['device_used_mib']) for r in train_gpu), default=None),
    'memory_scope': profile['memory_scope'],
    'jax_memory_fraction': profile.get('jax_memory_fraction', 'default'),
    'max_process_ram_gib': max((r.get('usage/psutil/proc_ram_gb', 0) for r in logs), default=None),
    'training_gpu_utilization_mean_pct': statistics.mean(float(r['gpu_util_pct']) for r in train_gpu) if train_gpu else None,
    'max_temperature_c': max(float(r['temperature_c']) for r in gpu),
    'post_warmup_windows': windows,
    'excluded_profiler_log_steps': sorted(profiler_steps),
    'all_recorded_losses_finite': (all(math.isfinite(v) for row in train
        for key, v in row.items() if key.startswith('train/loss/') or key == 'train/opt/loss') if train else None),
    'estimation_limits': 'Short-run engineering extrapolation, not a confidence interval. '
        'Does not measure later replay growth, long-run drift, extra evaluation, extension overhead, '
        'or seed-to-seed variability. GPU memory is whole-device sampled usage.'
}
if windows:
    fps = sum(w['steps'] for w in windows) / sum(w['elapsed_s'] for w in windows)
    low, high = min(w['env_fps'] for w in windows), max(w['env_fps'] for w in windows)
    startup = profile['training_start_s'] or 0
    summary.update(measured_env_steps_per_second=fps,
                   observed_window_fps_range=[low, high])
    for budget in [1000000, 1100000]:
        hours = (budget / fps + startup) / 3600
        summary[f'estimate_{budget}_steps'] = {
            'one_run_hours': hours, 'three_seeds_hours': hours * 3,
            'five_seeds_hours': hours * 5,
            'baseline_plus_extension_three_seeds_hours': hours * 6,
            'baseline_plus_extension_five_seeds_hours': hours * 10,
            'observed_window_range_one_run_hours': [(budget / high + startup) / 3600,
                                                   (budget / low + startup) / 3600]}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
axes[0].plot([float(r['elapsed_s']) for r in gpu],
             [float(r['device_used_mib']) / 1024 for r in gpu], label='Whole GPU used')
axes[0].axhline(baseline / 1024, color='gray', linestyle='--', label='Before run')
if profile['training_start_s'] is not None:
    axes[0].axvline(profile['training_start_s'], color='green', linestyle=':', label='Training loop starts')
axes[0].set(xlabel='Wall time (s)', ylabel='Used GPU memory (GiB)', title='Crafter real model: GPU sampling at ~1 s intervals')
axes[0].legend()
if windows:
    axes[1].plot([w['end_step'] for w in windows], [w['env_fps'] for w in windows], 'o-')
    axes[1].set(xlabel='Environment step', ylabel='Environment steps / second', title='Training windows after first loss log')
else:
    axes[1].text(.5, .5, 'No post-warmup training throughput available', ha='center', transform=axes[1].transAxes)
for ax in axes:
    ax.grid(alpha=.25)
fig.savefig(root / 'performance.png', dpi=160)
print(json.dumps(summary, indent=2))
