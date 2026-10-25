# Crafter batch karşılaştırması

Aynı 165.965.075 parametreli model, seed 0, dizi uzunluğu 64 ve train ratio 512 kullanıldı.
JAX bellek sınırı yüzde 90, preallocation kapalı. Model/loss kodu değiştirilmedi.
Her pilot replay dolumundan sonra yaklaşık 192 ortam adımlık eğitime ayrıldı.
İlk loss aralığı ve otomatik JAX profiler etkisi olan log aralıkları hız hesabından çıkarıldı.

| Batch | GPU tepe GiB | Başlangıca göre ek GiB | Ortam adımı/s | 1M saat | 6 koşu saat |
|---:|---:|---:|---:|---:|---:|
| 16 | 15.49 | 14.11 | 2.912 | 95.4 | 572.5 |
| 12 | 9.38 | 8.26 | 2.821 | 98.5 | 590.9 |
| 8 | 9.39 | 8.34 | 2.508 | 110.8 | 664.7 |
| 4 | 9.39 | 8.26 | 1.823 | 152.4 | 914.2 |

Pilot ölçümlerine göre önerilen batch: **12**. Seçim ölçütü: örneklenen tepede en az 2 GiB boş GPU belleği bırakanlar arasında en yüksek eğitim hızı.
Bu ayarda iki yöntem x üç seed için 590.9 GPU saati (24.6 kesintisiz gün); yüzde 20 planlama payıyla 709.1 saat / 29.5 gün.
İki yöntem x beş seed tahmini 984.9 GPU saati. Üç seed koşu sayısını ve varsayılan toplam maliyeti yüzde 40 azaltır.

GPU değerleri masaüstü dahil aygıt toplamıdır ve yaklaşık saniyede bir örneklenmiştir. Başlangıç değerleri pilotlar arasında farklıdır; ek GiB sütunu bu farkı görünür kılar.
Hız sonuçları kısa pilotlara aittir. Süre tahminleri extension adım maliyetinin baseline ile benzer olduğunu varsayar; yüzde 20 pay ölçülmüş bir ek maliyet değildir. Bu testler oyun skoru ya da seed değişkenliği karşılaştırması değildir.

Batch küçültmek VRAM yükünü azaltabilir, ancak replay kapasitesinin RAM maliyetini azaltmaz. Mevcut replay kodunda bir milyon görüntü ve latent kaydı yaklaşık 49,6 GiB ham veri eder; WSL yaklaşık 15,45 GiB RAM görüyor. Tam koşu öncesinde host RAM veya replay kapasitesi ayrı ele alınmalıdır; değişen ayarlar baseline ve extension için eşleştirilmelidir.

Tüm pilotlar adım sınırında, çıkış kodu 0 ile tamamlandı. Bir milyon adımlık eğitim başlatılmadı.
