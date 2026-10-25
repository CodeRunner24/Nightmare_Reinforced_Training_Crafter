"""Compare completed short pilots from saved telemetry; does not train."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
entries = [(16, 'crafter_real_profile_02'), (12, 'crafter_batch12_profile_01'), (8, 'crafter_batch8_profile_01'),
           (4, 'crafter_batch4_profile_01')]
rows = []
for batch, name in entries:
    data = json.loads((ROOT / 'logs' / name / 'summary.json').read_text())
    assert data['returncode'] == 0 and not data['timed_out'], name
    assert data['all_recorded_losses_finite'], name
    assert len(data['post_warmup_windows']) >= 3, name
    config = (ROOT / 'logs' / name / 'config.yaml').read_text()
    assert 'train_ratio: 512' in config and f'batch_size: {batch}' in config
    rows.append({
        'batch_size': batch, 'directory': name,
        'peak_whole_gpu_gib': data['peak_device_mib'] / 1024,
        'baseline_gpu_gib': data['baseline_device_mib'] / 1024,
        'peak_above_baseline_gib': data['peak_above_baseline_mib'] / 1024,
        'whole_gpu_free_at_peak_mib': 16303 - data['peak_device_mib'],
        'env_steps_per_second': data['measured_env_steps_per_second'],
        'observed_window_min_fps': data['observed_window_fps_range'][0],
        'observed_window_max_fps': data['observed_window_fps_range'][1],
        'speed_windows': len(data['post_warmup_windows']),
        'measured_post_warmup_steps': sum(w['steps'] for w in data['post_warmup_windows']),
        'measured_post_warmup_seconds': sum(w['elapsed_s'] for w in data['post_warmup_windows']),
        'total_wall_seconds': data['wall_seconds'],
        'startup_seconds': data['startup_to_training_seconds'],
        'one_million_steps_hours': data['estimate_1000000_steps']['one_run_hours'],
        'six_runs_hours': data['estimate_1000000_steps']['baseline_plus_extension_three_seeds_hours'],
        'ten_runs_hours': data['estimate_1000000_steps']['baseline_plus_extension_five_seeds_hours'],
    })
out = ROOT / 'logs' / 'batch_comparison'
out.mkdir(exist_ok=True)
(out / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
# Prefer throughput while leaving at least 2 GiB of sampled whole-device
# headroom. This is a planning heuristic, not a guarantee for longer runs.
candidates = [r for r in rows if r['whole_gpu_free_at_peak_mib'] >= 2048]
recommended = max(candidates, key=lambda r: r['env_steps_per_second'])
report = ['# Crafter batch karşılaştırması', '',
    'Aynı 165.965.075 parametreli model, seed 0, dizi uzunluğu 64 ve train ratio 512 kullanıldı.',
    'JAX bellek sınırı yüzde 90, preallocation kapalı. Model/loss kodu değiştirilmedi.',
    'Her pilot replay dolumundan sonra yaklaşık 192 ortam adımlık eğitime ayrıldı.',
    'İlk loss aralığı ve otomatik JAX profiler etkisi olan log aralıkları hız hesabından çıkarıldı.', '',
    '| Batch | GPU tepe GiB | Başlangıca göre ek GiB | Ortam adımı/s | 1M saat | 6 koşu saat |',
    '|---:|---:|---:|---:|---:|---:|']
for r in rows:
    report.append(f"| {r['batch_size']} | {r['peak_whole_gpu_gib']:.2f} | "
        f"{r['peak_above_baseline_gib']:.2f} | {r['env_steps_per_second']:.3f} | "
        f"{r['one_million_steps_hours']:.1f} | {r['six_runs_hours']:.1f} |")
report += ['',
    f"Pilot ölçümlerine göre önerilen batch: **{recommended['batch_size']}**. "
    'Seçim ölçütü: örneklenen tepede en az 2 GiB boş GPU belleği bırakanlar arasında en yüksek eğitim hızı.',
    f"Bu ayarda iki yöntem x üç seed için {recommended['six_runs_hours']:.1f} GPU saati "
    f"({recommended['six_runs_hours']/24:.1f} kesintisiz gün); yüzde 20 planlama payıyla "
    f"{recommended['six_runs_hours']*1.2:.1f} saat / {recommended['six_runs_hours']*1.2/24:.1f} gün.",
    f"İki yöntem x beş seed tahmini {recommended['ten_runs_hours']:.1f} GPU saati. "
    'Üç seed koşu sayısını ve varsayılan toplam maliyeti yüzde 40 azaltır.', '',
    'GPU değerleri masaüstü dahil aygıt toplamıdır ve yaklaşık saniyede bir örneklenmiştir. '
    'Başlangıç değerleri pilotlar arasında farklıdır; ek GiB sütunu bu farkı görünür kılar.',
    'Hız sonuçları kısa pilotlara aittir. Süre tahminleri extension adım maliyetinin baseline ile '
    'benzer olduğunu varsayar; yüzde 20 pay ölçülmüş bir ek maliyet değildir. '
    'Bu testler oyun skoru ya da seed değişkenliği karşılaştırması değildir.', '',
    'Batch küçültmek VRAM yükünü azaltabilir, ancak replay kapasitesinin RAM maliyetini azaltmaz. '
    'Mevcut replay kodunda bir milyon görüntü ve latent kaydı yaklaşık 49,6 GiB ham veri eder; '
    'WSL yaklaşık 15,45 GiB RAM görüyor. Tam koşu öncesinde host RAM veya replay kapasitesi '
    'ayrı ele alınmalıdır; değişen ayarlar baseline ve extension için eşleştirilmelidir.', '',
    'Tüm pilotlar adım sınırında, çıkış kodu 0 ile tamamlandı. Bir milyon adımlık eğitim başlatılmadı.']
(out / 'REPORT_TR.md').write_text('\n'.join(report) + '\n')
with (out / 'comparison.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

labels = [str(r['batch_size']) for r in rows]
colors = ['#9b5b52', '#7b629c', '#247a96', '#438b57']
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
values = [[r['peak_whole_gpu_gib'] for r in rows],
          [r['env_steps_per_second'] for r in rows],
          [r['six_runs_hours'] / 24 for r in rows]]
titles = ['Peak whole-GPU usage', 'Post-warmup training throughput', 'Estimated 6 runs x 1M steps']
ylabels = ['GiB (includes desktop)', 'Environment steps / second', 'Continuous single-GPU days']
for axis, vals, title, ylabel in zip(axes, values, titles, ylabels):
    bars = axis.bar(labels, vals, color=colors)
    axis.bar_label(bars, fmt='%.2f', padding=4)
    axis.set(title=title, xlabel='Batch size', ylabel=ylabel)
    axis.set_ylim(0, max(vals) * 1.18)
    axis.grid(axis='y', alpha=.2)
axes[0].axhline(16303 / 1024, color='black', linestyle='--', linewidth=1)
fig.suptitle('Same real Crafter model, sequence 64, train ratio 512, seed 0\nShort pilot measurements; runtime estimates are not full-run results', fontsize=11)
fig.savefig(out / 'comparison.png', dpi=170)
print('Batch | GPU peak GiB | Added GiB | Env steps/s | 1M hours | 6 runs hours')
for row in rows:
    print(f"{row['batch_size']:5} | {row['peak_whole_gpu_gib']:12.2f} | "
          f"{row['peak_above_baseline_gib']:9.2f} | {row['env_steps_per_second']:11.3f} | "
          f"{row['one_million_steps_hours']:8.1f} | {row['six_runs_hours']:12.1f}")
print(json.dumps(rows, indent=2))
