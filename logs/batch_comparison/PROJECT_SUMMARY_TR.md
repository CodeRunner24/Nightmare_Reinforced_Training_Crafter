# Nightmare Training: Proje Katkısı ve Deney Özeti

## 1. Ana fikir

Bu projenin ana katkısı, DreamerV3'ün hayal kurma (imagined rollout)
başlangıç durumlarını seçme biçimini değiştirmektir.

DreamerV3, hayal edilen gelecekleri çoğunlukla replay buffer'dan alınan
normal eğitim durumlarından başlatır. Önerilen **Nightmare Training**
yönteminde ise hayal rollout'larının sabit bir bölümü tehlikeli durumlardan
başlatılır:

- gerçek episode termination veya ölümden hemen önceki replay durumları,
- modelin episode'un yakında biteceğini tahmin ettiği durumlar.

Bu sayede ajan, yalnızca sık ve kolay durumları değil, hayatta kalmayı
tehdit eden anları da hayalinde daha sık görür. Hipotez, bu hedefli
imagination dağılımının ajanı tehlikeli durumları önceden tanımaya ve
ölümden kaçınmaya yönlendireceğidir.

## 2. DreamerV3'ten farkı

Değişiklik, DreamerV3'ün tüm modelini yeniden tasarlamak değildir. Adil
karşılaştırma için aşağıdaki bileşenler aynı tutulmalıdır:

| Bileşen | Baseline | Nightmare Training |
|---|---|---|
| Ortam | Crafter | Crafter |
| Model mimarisi | Aynı | Aynı |
| Optimizer ve loss'lar | Aynı | Aynı |
| Batch size | Aynı | Aynı |
| Replay kapasitesi | Aynı | Aynı |
| Gerçek çevre adımı | Aynı | Aynı |
| Seed'ler | Aynı | Aynı |
| Hayal rollout bütçesi | Aynı | Aynı |
| Değişen özellik | Normal start-state seçimi | Tehlikeli start-state'lerin sabit oranı |

Bu tasarımda performans farkı; daha büyük modelden, daha fazla gerçek
veriden veya daha fazla hesaplamadan değil, imagination başlangıçlarının
hedeflenmesinden kaynaklanır.

## 3. Araştırma iddiası

Projenin temel araştırma sorusu şudur:

> DreamerV3'te hayal rollout'larının bir bölümünü ölümden veya episode
> termination'dan önceki tehlikeli durumlardan başlatmak, aynı gerçek çevre
> etkileşim bütçesiyle Crafter'da performansı ve hayatta kalmayı artırır mı?

Başarı yalnızca skorla değil, mekanizmayı doğrudan ölçen metriklerle de
değerlendirilmelidir:

- episode skoru,
- episode uzunluğu,
- ilk ölüme kadar geçen adım sayısı,
- ölümle biten episode oranı,
- episode başına ölüm sayısı,
- modelin episode termination tahminleri.

Eğer Nightmare Training, baseline'dan daha yüksek skorla birlikte daha uzun
hayatta kalma süresi ve daha düşük ölüm oranı üretirse, sonuç önerilen
mekanizmayı destekler. Skor artmaz fakat hayatta kalma metrikleri iyileşirse,
yöntemin davranışı değiştirdiği ancak 100K adım bütçesinin skor farkını
göstermek için yetersiz kaldığı söylenmelidir.

## 4. Deney planı

### 4.1. 20K pilot

20K adımlık pilot, ana sonuç değil, fizibilite ve uygulama doğrulamasıdır.
Bir Crafter ortamında baseline ve Nightmare Training koşuları ile:

- kodun çalışması,
- tehlikeli durumların gerçekten seçilmesi,
- log ve checkpoint üretimi,
- GPU ve sistem RAM kullanımı,
- replay büyümesi,
- yaklaşık çalışma süresi

kontrol edilir.

### 4.2. 100K ana karşılaştırma

Ana deneyde baseline ve Nightmare Training aynı Crafter ayarlarıyla,
tercihen aynı üç seed'in eşleştirilmiş kullanımıyla 100.000 gerçek çevre
adımı çalıştırılmalıdır:

- Baseline: 100K adım, seed 0/1/2
- Nightmare Training: 100K adım, seed 0/1/2

Bu toplam altı koşudur. 20K pilot, 100K ana karşılaştırmanın yerine
geçmez. Üç seed mümkün değilse bu durum raporda açıkça belirtilmeli ve
baseline ile extension arasında seed veya ayar farkı yapılmamalıdır.

## 5. Ön ölçülen donanım ve batch sonuçları

Ölçümler, RTX 5070 Ti üzerinde, 16.303 MiB toplam GPU belleği görülen
ortamda, Crafter 5.000 adımlık gerçek model koşularından alınmıştır.
JAX GPU preallocation kapalıdır. WSL yaklaşık 15,45 GiB sistem RAM'i
görmektedir.

| Batch | 5K toplam süre | 100K doğrusal tahmin | Tepe GPU | Tepe süreç ağacı RSS | Minimum WSL kullanılabilir RAM |
|---:|---:|---:|---:|---:|---:|
| 12 | 29,17 dk | 9 saat 43 dk | 10,05 GiB | 8,21 GiB | 4,08 GiB |
| **14** | **25,91 dk** | **8 saat 38 dk** | **10,97 GiB** | **8,16 GiB** | **4,08 GiB** |
| 16 | 30,55 dk | 10 saat 11 dk | 15,58 GiB | 8,61 GiB | 3,74 GiB |

Batch 14 kısa pilotlarda en iyi dengeyi göstermiştir: batch 12'den daha
hızlıdır ve batch 16 gibi GPU belleğinin sınırına yaklaşmaz. Bu nedenle
ana deney için mevcut ölçümlere dayalı öneri **batch 14**'tür. Baseline ve
Nightmare Training aynı batch ayarını kullanmalıdır.

Bu süreler 5K pilotlardan türetilen tahminlerdir; uzun koşuda garanti
değildir. Checkpoint, derleme, sistem yükü ve olası duraklamalar için bir
100K koşusuna yaklaşık **9--10 saat** planlanmalıdır. Baseline ve extension
iki koşu olarak yaklaşık **18--20 saat**, üç seed ile toplam yaklaşık
**54--60 saat** planlama süresi gerekir.

## 6. Replay ve RAM sınırlaması

Final checkpoint ölçümünde replay geçişi başına ham alan maliyeti:

- görüntü: `64 x 64 x 3 uint8` = 12.288 byte,
- deterministic latent: `8192 float32` = 32.768 byte,
- stochastic latent: `32 x 64 float32` = 8.192 byte,
- diğer alanlarla birlikte ölçülen toplam: **53.279 byte/geçiş**.

Bu yaklaşık olarak:

- 100K geçişte **5,33 GB ham replay alanı**,
- 1M geçişte **53,28 GB ham replay alanı**

demektir. Gerçek süreç, chunk ve Python/NumPy ek yükleri nedeniyle daha
fazla bellek kullanabilir. 5K koşularında replay yaklaşık 0,305 GiB
olarak ölçülmüştür.

Bu nedenle batch size GPU VRAM'ini etkiler; replay'in sistem RAM'i
maliyetini çözmez. WSL limitini artırmak daha fazla RAM sağlayabilir,
ancak replay'in geçiş başına büyümesini değiştirmez. Replay kapasitesi
değiştirilirse baseline ve Nightmare Training'de aynı kapasite kullanılmalı
ve bu değişiklik raporda açıkça belirtilmelidir.

## 7. Paper sonucu ile karşılaştırma

DreamerV3 makalesindeki uzun eğitim sonucu dış referans olarak
kullanılabilir; ancak bu değer bizim 100K baseline sonucumuz değildir.

Üç ayrı sonuç raporlanmalıdır:

1. **Published DreamerV3 reference:** makaledeki referans koşul ve adım
   sayısı.
2. **Local DreamerV3 baseline:** bizim Crafter ortamımızda, örneğin 100K
   adımda çalıştırdığımız resmi baseline.
3. **Nightmare Training:** aynı 100K adım, aynı seed'ler ve aynı kaynak
   ayarlarıyla çalıştırdığımız extension.

100K Nightmare Training, makaledeki uzun eğitim skoruna ulaşır veya geçerse,
bu sonuç şu şekilde dikkatli raporlanabilir:

> Nightmare Training reached or exceeded the published DreamerV3 reference
> score using 100K environment steps rather than the longer reference
> training budget. This suggests improved sample efficiency under the
> tested Crafter protocol.

Bu ifade, doğrudan “DreamerV3'ü kesin olarak geçtik” demekten daha
doğrudur. Sıkı karşılaştırma için Crafter sürümü, skor tanımı, evaluation
protokolü, seed sayısı ve güven aralıkları aynı raporda verilmelidir.
100K baseline'ın skoru önceden bilinen bir değer değildir; deneyle
ölçülmelidir.

## 8. Proposal için kısa katkı özeti

> This project proposes Nightmare Training, a targeted modification to the
> imagination-start distribution of DreamerV3. Instead of starting all
> imagined rollouts from ordinary replay states, a fixed fraction starts
> from states shortly before death or episode termination. The world model,
> architecture, optimizer, replay capacity, real-environment budget and
> imagination budget remain unchanged. We will test the method in Crafter
> with a 20K feasibility pilot and a 100K baseline-versus-extension
> comparison. The main outcome is episode score, complemented by survival
> metrics such as episode length, steps to first death and the rate of
> death-ending episodes. Preliminary 5K measurements on an RTX 5070 Ti
> indicate that batch 14 is the best measured setting: it used 10.97 GiB
> peak GPU memory and completed in 25.91 minutes. Replay memory remains the
> main system-RAM constraint and will be kept identical across methods.

## 9. Mevcut durum ve açık işler

Tamamlananlar:

- Crafter kurulumu ve ortam doğrulaması,
- gerçek DreamerV3 modelinin GPU'da çalışması,
- loss ve parametre güncellemelerinin gözlenmesi,
- batch 12, 14 ve 16 için 5K kaynak pilotları,
- checkpoint ve replay bellek ölçümleri.

Henüz yapılmayanlar:

- 100K baseline koşuları,
- Nightmare Training kodunun uygulanması,
- 100K baseline--extension karşılaştırması,
- üç seed'in tamamı,
- makaledeki uzun eğitim sonucunun yeniden üretimi,
- 100K replay RAM fizibilitesinin uzun koşuda doğrulanması.

Bu nedenle mevcut ölçümler yöntemin nihai skor katkısını henüz kanıtlamaz;
donanım, hız ve replay maliyeti için hazırlık kanıtı sağlar. Nihai katkı
iddiası, aynı 100K bütçesinde baseline ve Nightmare Training sonuçları
alındıktan sonra yapılmalıdır.
