# CNN Hiperparametre Optimizasyonu ile DAGM 2007 Yüzey Kusuru Sınıflandırması

Bu depo, DAGM 2007 yüzey kusuru görüntüleri üzerinde dört CNN mimarisinin iki farklı hiperparametre optimizasyon yöntemiyle karşılaştırıldığı deneysel bir ön çalışmayı içerir. Çalışmanın ana değerlendirme fazı, aşırı öğrenme önlemleri eklendikten sonra yürütülen **Genetik Algoritma (GA)** ve **Bayesyen TPE** karşılaştırmasıdır. `repulsive_hybrid` klasöründeki çıktılar sonraki bir deneme fazına aittir; bu deneme ana sonuç ve çıkarım tablosuna dahil edilmemiştir.

Metindeki sayısal sonuçlar `results/ga/*_final_report.txt`, `results/bayesian/*_final_report.txt` ve `results/final_comparison_table.csv` dosyalarından alınmıştır. Kodun fiilen uygulamadığı bir mimari bileşen veya hiperparametre deneysel sonuç gibi sunulmamıştır.

## Özet

Bu çalışmada endüstriyel yüzey dokularını temsil eden DAGM 2007 veri seti üzerinde 10 sınıflı görüntü sınıflandırma problemi ele alınmıştır. Deneylerde iki transfer öğrenme modeli (VGG16, ResNet50) ve iki sıfırdan eğitilen CNN mimarisi (Custom CNN, Proposed CNN/Custom V2) kullanılmıştır. Hiperparametre seçimi için Genetik Algoritma ve Bayesyen TPE karşılaştırılmış; öğrenme oranı, dropout, batch size, optimizer etiketi ve modele özgü mimari parametreler arama uzayına dahil edilmiştir.

Ana deney fazında en yüksek macro F1 değeri Proposed CNN + Bayesyen TPE koşulunda 0.8842 olarak ölçülmüştür. VGG16 için GA ve Bayesyen TPE sonuçları birbirine çok yakındır. ResNet50 ve Proposed CNN koşullarında Bayesyen TPE daha güçlü sonuç verirken, Custom CNN koşulunda GA daha yüksek test başarımı üretmiştir. Sınıf bazlı raporlar, genel accuracy değerlerinin bazı sınıflardaki zayıf performansı örtebildiğini göstermektedir.

## Giriş

Amaç, aynı veri hattı ve aynı düzenlileştirme stratejileri altında, önceden eğitilmiş transfer öğrenme modelleri ile sıfırdan eğitilen CNN modellerinde GA ve Bayesyen TPE optimizasyonunun sınıflandırma başarımını karşılaştırmaktır. Deneysel çıkarım şu soruya odaklanır:

> Hazır ağırlıklarla başlatılan daha düzgün optimizasyon yüzeylerinde Bayesyen TPE, sıfırdan eğitilen mimarilerde ise GA daha avantajlı mıdır?

Problem, 512x512 gri tonlamalı yüzey görüntülerini `Class1` ile `Class10` arasındaki 10 sınıftan birine atamaktır. Sınıflar veri kümesinde semantik kusur adlarıyla değil, sınıf numaralarıyla temsil edilmektedir. Bu nedenle çalışmanın odağı kusur tiplerini isimlendirmek değil, sınıflar arası doku örüntülerini ayırt eden CNN modellerinin hangi optimizasyon stratejisiyle daha kararlı sonuç verdiğini incelemektir.

Mevcut sonuçlar yukarıdaki çıkarımı kısmen desteklemektedir. ResNet50 ve Proposed CNN (`CUSTOM_V2`) için Bayesyen TPE daha yüksek test başarımı vermiştir. Custom CNN için GA açık biçimde daha başarılıdır. VGG16'da iki yöntem birbirine çok yakındır ve GA çok küçük bir farkla daha yüksek accuracy üretmiştir. Bu nedenle sonuçlar, güçlü bir evrensel kuraldan çok, mimari ve uygulama ayrıntılarına bağlı deneysel bir eğilim olarak yorumlanmalıdır.

## Proje Fazları

1. **İlk karşılaştırma fazı:** GA ve Bayesyen TPE daha sınırlı model kümesi üzerinde denenmiştir.
2. **Ana deney fazı:** Aşırı öğrenmeyi azaltan veri artırma, label smoothing, weight decay, dropout ve early stopping stratejileri eklendikten sonra dört model iki optimizasyon yöntemiyle karşılaştırılmıştır. Bu README'de raporlanan sonuçlar bu faza aittir.
3. **Repulsive-hybrid deneme fazı:** GA ile hafif topografya taraması yapıp Bayesyen TPE'yi yönlendirme fikri denenmiştir. Mevcut çıktılar beklenen iyileşmeyi sağlamadığı için ana bulgulara dahil edilmemiştir.

## Veri Seti ve Veri Hattı

Veri dizini: `data/DAGM2007`

Kodda kullanılan ayrımlar:

| Ayrım | Görüntü sayısı |
| --- | ---: |
| Eğitim klasörü | 9.660 |
| Eğitimde kullanılan stratified subset | 500 |
| Doğrulama | 2.412 |
| Test | 2.418 |

Veri seti yerel dizinde önceden `train`, `val` ve `test` olarak ayrılmıştır; kod bu ayrımı yeniden üretmez, doğrudan mevcut klasörleri okur. Eğitim klasöründe sınıflar dengeli değildir; bazı sınıflarda 805, bazı sınıflarda 1.610 görüntü bulunur. Deneysel karşılaştırmada eğitim yükünü ve sınıf dengesizliğinin etkisini sınırlamak için eğitimde her sınıftan en fazla 50 örnek seçilmektedir. Bu tercih `src/utils/data_loader.py` içinde `samples_per_class = 50` olarak tanımlıdır.

Sınıf düzeyinde yerel veri dağılımı:

| Sınıf | Eğitim klasörü | Eğitimde kullanılan | Doğrulama | Test |
| --- | ---: | ---: | ---: | ---: |
| Class1 | 805 | 50 | 172 | 173 |
| Class10 | 1.610 | 50 | 345 | 345 |
| Class2 | 805 | 50 | 172 | 173 |
| Class3 | 805 | 50 | 172 | 173 |
| Class4 | 805 | 50 | 172 | 173 |
| Class5 | 805 | 50 | 172 | 173 |
| Class6 | 805 | 50 | 172 | 173 |
| Class7 | 1.610 | 50 | 345 | 345 |
| Class8 | 1.610 | 50 | 345 | 345 |
| Class9 | 1.610 | 50 | 345 | 345 |

Test raporlarında sınıf sırası `ImageFolder` sıralamasından geldiği için şu şekildedir:

`Class1, Class10, Class2, Class3, Class4, Class5, Class6, Class7, Class8, Class9`

Örnek görüntüler 512x512 gri tonlamalı PNG dosyalarıdır. Model girişinde tüm görüntüler 224x224 boyutuna ölçeklenir ve 3 kanallı RGB tensörüne dönüştürülür. Bu dönüşüm özellikle VGG16 ve ResNet50 için gereklidir; çünkü bu modeller ImageNet üzerinde 3 kanallı girişlerle eğitilmiş ağırlıklarla başlatılmaktadır.

Eğitim dönüşümleri:

- `Resize((224, 224))`
- `Grayscale(num_output_channels=3)`
- `RandomHorizontalFlip`
- `RandomVerticalFlip`
- `RandomRotation(15)`
- `ColorJitter(brightness=0.3, contrast=0.3)`
- `ToTensor`
- `AddGaussianNoise(mean=0, std=0.05)`
- `RandomErasing(p=0.5)`
- ImageNet ortalama ve standart sapması ile normalizasyon

Doğrulama ve test dönüşümleri yalnızca yeniden boyutlandırma, 3 kanala dönüştürme, tensöre çevirme ve normalizasyon içerir.

## Deney Tasarımı

Karşılaştırmanın temel birimi, aynı veri hattı üzerinde eğitilen bir model-yöntem çiftidir. Model tarafında VGG16, ResNet50, Custom CNN ve Proposed CNN; optimizasyon tarafında ise GA ve Bayesyen TPE değerlendirilmiştir. Test başarımı final eğitim tamamlandıktan sonra ayrı test setinde ölçülmüştür.

Karşılaştırmanın adil kalması için görüntü boyutu, sınıf sayısı, eğitim subset kuralı, veri artırma hattı, kayıp fonksiyonu, weight decay, learning-rate scheduler ve early stopping stratejisi ortak tutulmuştur. Değişen bileşenler model mimarisi ve hiperparametre optimizasyon yöntemidir. Optimizasyon yöntemleri öğrenme oranı, dropout, batch size, optimizer etiketi ve sıfırdan eğitilen modeller için mimari genişlik/derinlik parametrelerini seçmektedir.

Başarım, tek başına accuracy ile değil; macro precision, macro recall, macro F1 ve confusion matrix çıktılarıyla birlikte değerlendirilmiştir. Bunun nedeni test setinde sınıf desteklerinin eşit olmaması ve bazı modellerin yüksek genel doğruluğa rağmen belirli sınıflarda düşük recall üretebilmesidir.

## Eğitim Protokolü

Ortak eğitim ayarları:

| Bileşen | Değer |
| --- | --- |
| Kayıp fonksiyonu | `CrossEntropyLoss(label_smoothing=0.1)` |
| Öğrenme oranı planlayıcı | `CosineAnnealingLR(T_max=epochs)` |
| Weight decay | `1e-3` |
| Hiperparametre arama değerlendirmesi | 5 epoch, patience 3 |
| Final eğitim | 15 epoch, patience 5 |
| Görüntü boyutu | 224x224 |
| Sınıf sayısı | 10 |

Final eğitimde batch size, ilgili optimizasyon yönteminin seçtiği en iyi hiperparametrelerden alınmıştır.

`CrossEntropyLoss(label_smoothing=0.1)` çok sınıflı sınıflandırma problemine doğrudan uygundur ve label smoothing, modelin sınırlı eğitim subset'i üzerinde tek sınıfa aşırı güvenle bağlanmasını azaltmak için kullanılmıştır. `weight_decay=1e-3`, tüm optimizerlarda ağırlık büyüklüklerini cezalandırarak aşırı öğrenmeyi sınırlamaya yardımcı olur. `CosineAnnealingLR`, eğitim süresi boyunca öğrenme oranını kademeli olarak düşürerek final epoch'larda daha küçük güncellemeler yapılmasını sağlar. Early stopping doğrulama doğruluğu gelişmediğinde eğitimi keser ve en iyi doğrulama ağırlıklarına geri dönülmesini sağlar.

Aşırı öğrenmeyi azaltmak için kullanılan önlemler birlikte değerlendirilmelidir: sınıf başına 50 örnekle eğitim yapılması problemi zorlaştırır; buna karşılık random flip, rotation, brightness/contrast değişimi, Gauss gürültüsü ve random erasing modelin tekil görsel örnekleri ezberlemesini zorlaştırır. Dropout aralığının 0.2 ile 0.6 arasında aranması da sınıflandırıcı başlıklarında farklı düzenlileştirme düzeylerinin denenmesini sağlar.

## Model Mimarileri

### VGG16

Kaynak: `src/models/vgg16_model.py`

- Torchvision `vgg16` modeli ImageNet ağırlıklarıyla başlatılır.
- `model.features` içindeki tüm parametreler önce dondurulur.
- Kod `children[-fine_tune_layers:]` ile son 4 feature modülünü yeniden eğitime açar. VGG16 feature diziliminde bu aralık parametre içeren son iki konvolüsyon katmanını ve parametresiz aktivasyon/pooling modüllerini kapsar.
- Sınıflandırıcı başlığı şu yapı ile değiştirilir:
  `Linear(25088, dense_units) -> BatchNorm1d -> ReLU -> Dropout -> Linear(dense_units, 10)`

### ResNet50

Kaynak: `src/models/resnet50_model.py`

- Torchvision `resnet50` modeli ImageNet ağırlıklarıyla başlatılır.
- Tüm backbone parametreleri dondurulur.
- `layer4` içindeki son 2 residual blok eğitime açılır.
- Sınıflandırıcı başlığı:
  `Linear(2048, dense_units) -> BatchNorm1d -> ReLU -> Dropout -> Linear(dense_units, dense_units/2) -> ReLU -> Dropout(dropout/2) -> Linear(dense_units/2, 10)`

### Custom CNN

Kaynak: `src/models/custom_cnn_model.py`

- Model sıfırdan eğitilir.
- Stem:
  `Conv2d(3, base_filters, kernel=3, stride=2, padding=1, bias=False) -> BatchNorm2d -> LeakyReLU(0.3) -> MaxPool2d(2)`
- Ardından `num_blocks` adet depthwise-separable blok kullanılır.
- Her blokta depthwise `Conv2d(groups=in_channels)`, pointwise `Conv2d(1x1)`, `BatchNorm2d`, `LeakyReLU(0.3)` ve kanal uyumsuzluğu varsa `1x1` residual shortcut bulunur.
- Bloklar arasında, son blok hariç, `MaxPool2d(2)` uygulanır.
- Başlık:
  `AdaptiveAvgPool2d(1) -> Flatten -> Linear -> BatchNorm1d -> LeakyReLU(0.3) -> Dropout -> Linear(10)`
- Not: Dosyada `SEBlock` tanımlıdır; ancak mevcut `SepConvBlock.forward` yolunda kullanılmaz. Bu nedenle raporlanan Custom CNN sonuçları aktif SE-attention içeren bir model olarak yorumlanmamalıdır.

### Proposed CNN / Custom V2

Kaynak: `src/models/custom_cnn_v2_model.py`

- Model sıfırdan eğitilir.
- Acici vd. (2020) çalışmasındaki CNN tasarımından esinlenen 5 bloklu yapı kullanılır.
- Her blok:
  `Conv2d(kernel_size, stride=1, padding=kernel_size//2) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)`
- Kanal genişlikleri:
  `base_filters -> 2*base_filters -> 4*base_filters -> 4*base_filters -> 4*base_filters`
- Başlık:
  `AdaptiveAvgPool2d(1) -> Flatten -> Dropout -> Linear(10)`
- Bu modelde `get_optimizer` fonksiyonu arayüz uyumluluğu için `optimizer_name` parametresini alır; fakat fiilen her durumda `AdamW` döndürür. Bu nedenle `CUSTOM_V2` sonuçlarında seçilen optimizer adı değil, etkin optimizer olarak AdamW dikkate alınmalıdır.

## Optimizasyon Yöntemleri

### Genetik Algoritma

Kaynak: `src/optimization/genetic_algorithm.py`

GA, DEAP ile uygulanmıştır. Her birey tamsayı genlerden oluşur ve genler hiperparametre listelerine modulo ile çözümlenir.

Ortak arama uzayı:

| Hiperparametre | Aday değerler |
| --- | --- |
| Learning rate | `1e-5`, `1e-4`, `5e-4`, `1e-3`, `1e-2` |
| Dropout | `0.2`, `0.3`, `0.4`, `0.5`, `0.6` |
| Dense units | `128`, `256`, `384`, `512` |
| Optimizer etiketi | `adam`, `sgd`, `rmsprop` |
| Batch size | `16`, `32`, `64` |

Custom modeller için ek genler:

| Hiperparametre | Aday değerler |
| --- | --- |
| Base filters | `16`, `32`, `64` |
| Num blocks | `2`, `3`, `4`, `5` |
| Kernel size | `3`, `5` |

GA ayarları:

| Parametre | Değer |
| --- | ---: |
| Popülasyon | 10 |
| Nesil sayısı | 8 |
| Birey başına eğitim | 5 epoch |
| Seçilim | Tournament selection, `tournsize=3` |
| Çaprazlama | Two-point crossover, `cxpb=0.7` |
| Mutasyon | Uniform integer mutation, `mutpb=0.2`, `indpb=0.2` |

`CUSTOM_V2` için GA kodu `num_blocks` geni üretir; ancak model mimarisi sabit 5 blokludur. Bu gen final modelin blok sayısını değiştirmez.

GA arama uzayı ayrık değerlerden oluşacak şekilde tasarlanmıştır. Bunun nedeni DEAP bireylerinin tamsayı genlerle kodlanması ve her genin belirli bir hiperparametre listesindeki indekse karşılık gelmesidir. Öğrenme oranı seçenekleri `1e-5` ile `1e-2` arasında tutulmuştur; böylece hem transfer öğrenme modelleri için küçük güncellemeler hem de sıfırdan eğitilen modeller için daha agresif öğrenme oranları denenebilmektedir. Batch size seçenekleri 16, 32 ve 64 ile sınırlıdır; bu aralık hem GPU bellek kullanımını kontrol altında tutar hem de küçük batch kaynaklı düzenlileştirme etkisinin denenmesine izin verir.

Popülasyonun 10, nesil sayısının 8 seçilmesi toplam hesaplama maliyetini sınırlayan bir deneysel bütçe oluşturur. Her birey 5 epoch eğitildiği için GA, final eğitimden önce hızlı fakat karşılaştırılabilir bir doğrulama skoru üretir. Tournament selection, güçlü bireylerin korunmasını sağlarken iki noktalı çaprazlama ve uniform mutasyon arama uzayında farklı hiperparametre kombinasyonlarının denenmesini sağlar. Mutasyon olasılığının 0.2 olması, popülasyonun tamamen rastgele dağılmadan yeni bölgeleri taramasına olanak verir.

### Bayesyen TPE

Kaynak: `src/optimization/bayesian_tpe.py`

Bayesyen optimizasyon Optuna `TPESampler(seed=42)` ile uygulanmıştır.

| Hiperparametre | Arama biçimi |
| --- | --- |
| Learning rate | Log-uniform, `1e-5` ile `1e-2` arası |
| Dropout | Sürekli, `0.2` ile `0.6` arası |
| Dense units | Kategorik: `128`, `256`, `384`, `512` |
| Optimizer etiketi | Kategorik: `adam`, `sgd`, `rmsprop` |
| Batch size | Kategorik: `16`, `32`, `64` |
| Custom base filters | Kategorik: `16`, `32`, `64` |
| Custom num blocks | Tamsayı: `2` ile `5` arası |
| Custom kernel size | Kategorik: `3`, `5` |
| Custom V2 base filters | Kategorik: `8`, `16`, `32` |
| Custom V2 kernel size | Kategorik: `3`, `5` |

Bayesyen TPE ayarları:

| Parametre | Değer |
| --- | ---: |
| Trial sayısı | 20 |
| Trial başına eğitim | 5 epoch |
| Erken durdurma | patience 3 |

Bayesyen TPE tarafında öğrenme oranı sürekli ve logaritmik ölçekte aranmıştır. Bu tercih, öğrenme oranının doğrusal değil çarpan ölçeğinde etkili olmasından dolayı uygundur; `1e-5` ile `1e-2` arasındaki aralık, çok küçük fine-tuning adımlarından hızlı öğrenmeye kadar geniş bir alanı kapsar. Dropout sürekli aralıkta aranırken dense units, batch size ve optimizer etiketi kategorik değişkenler olarak verilmiştir. Böylece TPE, iyi sonuç veren denemelerin çevresinde olasılıksal yoğunluk kurarak sonraki trial'ları yönlendirebilir.

Trial sayısı 20 ile sınırlıdır ve her trial 5 epoch eğitilir. Bu nedenle Bayesyen TPE sonuçları tam eğitim değil, doğrulama performansına dayalı kısa süreli aday değerlendirmeleridir. En iyi bulunan hiperparametreler daha sonra 15 epoch final eğitimde kullanılmış ve test metrikleri bu final modellerden alınmıştır.

## Final Hiperparametreler ve Model Büyüklükleri

Parametre sayıları final raporlardaki en iyi hiperparametrelerle model yeniden kurularak hesaplanmıştır.

| Model | Yöntem | Learning rate | Dropout | Batch | Kayıtlı optimizer | Etkin optimizer | Ek parametreler | Toplam parametre | Eğitilebilir parametre |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| VGG16 | GA | 0.0001 | 0.3000 | 16 | rmsprop | RMSprop | dense=512 | 27.566.410 | 15.211.530 |
| VGG16 | Bayesyen TPE | 0.000233 | 0.5141 | 32 | adam | AdamW | dense=384 | 24.353.482 | 11.998.602 |
| ResNet50 | GA | 0.0005 | 0.2000 | 32 | adam | AdamW | dense=256 | 24.067.274 | 9.484.426 |
| ResNet50 | Bayesyen TPE | 0.000233 | 0.5141 | 32 | adam | AdamW | dense=384 | 24.371.466 | 9.788.618 |
| Custom CNN | GA | 0.0100 | 0.3000 | 32 | adam | AdamW | base=64, blocks=4, kernel=5 | 677.066 | 677.066 |
| Custom CNN | Bayesyen TPE | 0.000082 | 0.2391 | 16 | sgd | SGD | base=64, blocks=5, kernel=3 | 2.648.266 | 2.648.266 |
| Proposed CNN | GA | 0.0100 | 0.3000 | 64 | adam | AdamW | base=32, kernel=5 | 1.080.330 | 1.080.330 |
| Proposed CNN | Bayesyen TPE | 0.007887 | 0.5234 | 16 | sgd | AdamW | base=32, kernel=5 | 1.080.330 | 1.080.330 |

## Test Sonuçları

Tüm değerler test seti üzerinde hesaplanmıştır. Precision, recall ve F1 değerleri macro ortalamadır.

| Model | Yöntem | Accuracy | Precision | Recall | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| VGG16 | GA | 0.9173 | 0.8538 | 0.8844 | 0.8649 |
| VGG16 | Bayesyen TPE | 0.9152 | 0.8488 | 0.8881 | 0.8655 |
| ResNet50 | GA | 0.8379 | 0.8088 | 0.8728 | 0.8234 |
| ResNet50 | Bayesyen TPE | 0.8710 | 0.9342 | 0.9081 | 0.8704 |
| Custom CNN | GA | 0.8288 | 0.8125 | 0.8688 | 0.8215 |
| Custom CNN | Bayesyen TPE | 0.6716 | 0.7051 | 0.6701 | 0.6198 |
| Proposed CNN | GA | 0.8180 | 0.8162 | 0.8183 | 0.7907 |
| Proposed CNN | Bayesyen TPE | 0.9189 | 0.9213 | 0.8922 | 0.8842 |

Bayesyen TPE değerlerinden GA değerleri çıkarıldığında oluşan farklar:

| Model | Accuracy farkı | Macro F1 farkı | Sayısal yorum |
| --- | ---: | ---: | --- |
| VGG16 | -0.0021 | +0.0006 | İki yöntem pratik olarak benzer düzeydedir. |
| ResNet50 | +0.0331 | +0.0470 | Bayesyen TPE daha güçlüdür. |
| Custom CNN | -0.1572 | -0.2017 | GA belirgin biçimde daha güçlüdür. |
| Proposed CNN | +0.1009 | +0.0935 | Bayesyen TPE belirgin biçimde daha güçlüdür. |

## Karşılaştırmalı Bulgular

VGG16 için GA ve Bayesyen TPE sonuçları birbirine çok yakındır. GA accuracy değerinde 0.0021 puan daha yüksektir; Bayesyen TPE ise macro F1 değerinde 0.0006 puan daha yüksektir. Bu farklar, VGG16 için iki yöntemin pratik olarak benzer düzeyde sonuç verdiğini göstermektedir.

ResNet50 için Bayesyen TPE daha güçlü sonuç üretmiştir. Accuracy 0.8379'dan 0.8710'a, macro F1 0.8234'ten 0.8704'e yükselmiştir. Bu modelde Bayesyen TPE'nin seçtiği daha yüksek dropout ve AdamW ayarı, test setinde daha dengeli macro metriklerle sonuçlanmıştır.

Custom CNN için GA belirgin biçimde daha başarılıdır. Accuracy 0.6716'dan 0.8288'e, macro F1 0.6198'den 0.8215'e yükselmiştir. Ancak sınıf bazında Class9 için GA raporunda precision, recall ve F1 değerleri 0.00'dır. Bu nedenle GA genel başarıyı artırmış olsa da sınıf düzeyinde tam dengeli bir model üretmemiştir.

Proposed CNN için Bayesyen TPE en yüksek genel sonucu vermiştir. Accuracy 0.9189 ve macro F1 0.8842 ile ana fazdaki en iyi macro F1 değerine ulaşmıştır. Buna karşın Class4 için Bayesyen TPE raporunda recall 0.10 ve F1 0.17'dir. Dolayısıyla bu modelin yüksek genel başarımı, tüm sınıflarda homojen başarı anlamına gelmemektedir.

Sınıf bazlı raporlar, yalnızca accuracy ile karar vermenin yanıltıcı olabileceğini göstermektedir. Örneğin VGG16'ın iki koşusunda da Class3 için precision, recall ve F1 0.00'dır. ResNet50-Bayesyen koşusunda Class9 recall değeri 0.11'dir. Bu bulgular, ana sonuçların macro F1 ve confusion matrix ile birlikte değerlendirilmesini gerektirir.

## Hybrid Model İçin Hazırlık Çalışması

Depoda `src/optimization/repulsive_bayesian.py` içinde GA ve Bayesyen TPE'yi birleştirmeye yönelik bir hazırlık denemesi bulunmaktadır. Bu kod iki fazlıdır. İlk fazda kısa bir GA çalıştırılır; `n_generations=2`, `pop_size=5` ve `epochs_per_eval=3` ile her optimizer tipi için doğrulama başarımına dayalı eşik değerleri çıkarılır. Kod bu eşikleri MAP-Elites yaklaşımından esinlenen basitleştirilmiş bir ızgara mantığıyla, mevcut haliyle optimizer tipine göre tutar. İkinci fazda Bayesyen TPE çalıştırılır ve GA fazından gelen eşikler, düşük performanslı trial'ları erken kesmek için `prune_threshold` olarak kullanılır.

Bu hybrid deneme mevcut ana sonuçların yerine geçirilmemiştir; çünkü eldeki çıktılar GA/Bayesyen TPE ana fazına göre beklenen iyileştirmeyi güvenilir biçimde göstermemektedir. Bu nedenle README'deki ana bulgular, overfitting önlemleri sonrası üretilen GA ve Bayesyen TPE final raporlarına dayandırılmıştır.

Çalışmanın sonraki aşaması için önerilen araştırma yönü, ders yürütücüsünün onayı halinde, GA'yı yalnızca en iyi hiperparametreyi arayan bir yöntem olarak değil, hiperparametre uzayının topografyasını çıkaran bir analiz aracı olarak kullanmaktır. Bu yönde mevcut kod genişletilerek ızgara yalnızca optimizer tipine göre değil; learning rate aralıkları, dropout bölgeleri, batch size, filtre sayısı, blok sayısı ve kernel size gibi eksenlere göre de ayrılabilir. Böylece GA ile düşük ve yüksek performans bölgeleri haritalanabilir; ardından Bayesyen TPE sıfırdan eğitilen mimariler için daha umut verici bölgelere yönlendirilebilir. Bu öneri mevcut dosyalardaki hybrid hazırlık fikriyle uyumludur, ancak mevcut ana deney sonuçlarının yerine geçen tamamlanmış bir sonuç olarak değerlendirilmemelidir.

## Görsel Çıktılar

Ana faza ait grafikler `results/plots` altında yer alır.

### VGG16

![VGG16 GA Confusion Matrix](results/plots/VGG16_GA_cm.png)
![VGG16 GA History](results/plots/VGG16_GA_history.png)
![VGG16 Bayesian Confusion Matrix](results/plots/VGG16_BAYESIAN_cm.png)
![VGG16 Bayesian History](results/plots/VGG16_BAYESIAN_history.png)

### ResNet50

![ResNet50 GA Confusion Matrix](results/plots/RESNET50_GA_cm.png)
![ResNet50 GA History](results/plots/RESNET50_GA_history.png)
![ResNet50 Bayesian Confusion Matrix](results/plots/RESNET50_BAYESIAN_cm.png)
![ResNet50 Bayesian History](results/plots/RESNET50_BAYESIAN_history.png)

### Custom CNN

![Custom CNN GA Confusion Matrix](results/plots/CUSTOM_GA_cm.png)
![Custom CNN GA History](results/plots/CUSTOM_GA_history.png)
![Custom CNN Bayesian Confusion Matrix](results/plots/CUSTOM_BAYESIAN_cm.png)
![Custom CNN Bayesian History](results/plots/CUSTOM_BAYESIAN_history.png)

### Proposed CNN

![Proposed CNN GA Confusion Matrix](results/plots/CUSTOM_V2_GA_cm.png)
![Proposed CNN GA History](results/plots/CUSTOM_V2_GA_history.png)
![Proposed CNN Bayesian Confusion Matrix](results/plots/CUSTOM_V2_BAYESIAN_cm.png)
![Proposed CNN Bayesian History](results/plots/CUSTOM_V2_BAYESIAN_history.png)

### Toplu Karşılaştırma

![Final Comparison](results/plots/final_comparison.png)

## Yeniden Üretilebilirlik

Kurulum:

```bash
pip install -r requirements.txt
```

Ana deney komutu:

```bash
python src/train_and_compare.py
```

Bu komut `CONFIG` içinde tanımlanan modellere ve optimizasyon yöntemlerine göre arama ve final eğitim fazlarını çalıştırır. Mevcut `results` klasöründeki ana README sonuçları, GA ve Bayesyen TPE final raporlarına dayalıdır.

## Dizin Yapısı

```text
src/
  models/
    vgg16_model.py
    resnet50_model.py
    custom_cnn_model.py
    custom_cnn_v2_model.py
    proposed_cnn_model.py
  optimization/
    genetic_algorithm.py
    bayesian_tpe.py
    repulsive_bayesian.py
  utils/
    data_loader.py
    trainer.py
    metrics.py
    plotter.py
  train_and_compare.py
results/
  ga/
  bayesian/
  repulsive_hybrid/
  plots/
data/
  DAGM2007/
```

## Sınırlılıklar

- Eğitim subset'i sınıf başına 50 görüntü ile sınırlandırılmıştır. Bu tercih yöntemler arası farkı görünür kılabilir; ancak tam veriyle elde edilecek sonuçları doğrudan temsil etmez.
- Random seed tüm veri alt örnekleme sürecinde sabitlenmemiştir. Bu nedenle deneyler tamamen deterministik değildir.
- `CUSTOM_V2` modelinde optimizer etiketi arama uzayında görünür; fakat etkin optimizer kodda her zaman AdamW'dir.
- `Custom CNN` dosyasında SE bloğu tanımlı olsa da mevcut ileri yayılımda kullanılmaz.
- Bazı modeller yüksek genel doğruluğa rağmen belirli sınıflarda zayıf sonuç vermektedir. Bu nedenle çalışma, yalnızca accuracy üzerinden değil macro F1 ve sınıf bazlı raporlarla değerlendirilmelidir.

## Referans

Acici, K., Beyaz, S., & Sumer, E. (2020). *Femoral neck fracture detection in X-ray images using deep learning and genetic algorithm approaches*. Joint Diseases and Related Surgery, 31(2), 175-183.
