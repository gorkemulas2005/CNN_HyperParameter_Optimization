# YZM304 Deep Learning - Proje 5: CNN Hiperparametre Optimizasyonu

**Hazırlayanlar:**
- Ulaş Görkem KAZAN (23291785)
- Bora DOĞRU (23291786)

**Proje Konusu:** Evrimsel Algoritmalar (Genetik Algoritma) ve Olasılıksal Yöntemler (Bayesyen TPE) ile CNN Hiperparametrelerinin Sınıflandırma Problemlerinde Optimizasyonu ve Karşılaştırmalı Analizi.

---

## 1. GİRİŞ (Introduction)

YZM304 Deep Learning dersi kapsamında 5. proje için hazırlanan bu çalışmada, karmaşık yüzey/doku hatalarının sınıflandırılmasında Konvolüsyonel Sinir Ağlarının (CNN) performansını maksimize etmek amaçlanmıştır. Standart "deneme-yanılma" (grid/random search) yöntemlerinin yüksek donanım ve zaman maliyeti yaratması problemini çözmek adına, Evrimsel (Genetik Algoritma) ve İstatistiksel/Olasılıksal (Bayesyen TPE) tabanlı akıllı hiperparametre optimizasyon teknikleri birbirleriyle yarışacak şekilde uygulanmıştır. Ayrıca bu çalışmada literatürde özgün bir yaklaşım olan "Repulsive Hybrid" (Genetik Algoritma ile kötü bölgelerin tespit edilip Bayesyen TPE'nin bu alanlardan uzaklaştırılması) algoritması kullanılarak arama uzayı optimize edilmiştir.

---

## 2. MATERYAL VE METOT (Method)

### 2.1. Veri Seti (DAGM 2007)
Çalışmada, endüstriyel yüzey kusurlarını tespit etmek için standart bir benchmark olan **DAGM 2007** (Deutsche Arbeitsgemeinschaft für Mustererkennung) veri seti kullanılmıştır.
- **Sınıf Sayısı:** 10 farklı yüzey/doku sınıfı.
- **Görsel Özellikleri:** 512x512 boyutlarında gri tonlamalı (grayscale) görüntüler; ağlar için 224x224 RGB formatına dönüştürülmüştür.
- **Dağılım:** Eğitim süresini ve modelin genelleme yeteneğini zorlaştırmak için her sınıftan 50 adet "stratified" (tabakalı) örneklem seçilmiş ve Gaussian gürültü eklenmiştir.

### 2.2. Kullanılan Modeller
Farklı uzay topolojilerine sahip dört mimari karşılaştırılmıştır:
1. **VGG16 (Transfer Learning):** Sıralı ve ardışık 3x3 konvolüsyonlarla güçlü lokal özellik çıkaran mimari.
2. **ResNet50 (Transfer Learning):** Residual (artık) bağlantılarla kaybolan gradyan problemini çözen derin mimari.
3. **Custom CNN:** Squeeze-and-Excitation (SE) dikkat mekanizması ve Depthwise Separable convolutions içeren özgün tasarım.
4. **Proposed CNN:** Koray Açıcı vd. (2020) çalışmasından referans alınarak tasarlanmış 5 bloklu parametrik ağ.

### 2.3. Optimizasyon Stratejisi
- **Genetik Algoritma:** Seçilim, çaprazlama (crossover) ve mutasyon operatörleri kullanılarak popülasyon bazlı global bir arama yapılmıştır.
- **Bayesyen TPE:** Tree-structured Parzen Estimator kullanılarak geçmiş denemelerden öğrenen olasılıksal bir dağılım yaratılmıştır.
- **Repulsive Hybrid (Önerilen):** Genetik Algoritma'nın lokal minimum ve kötü tuzakları (trap) bulması sağlanmış, ardından Optuna tabanlı Bayesyen TPE'nin buraları erken budaması (pruning) hedeflenmiştir.

---

## 3. BULGULAR (Results)

Yapılan deneysel çalışmaların test verisi üzerindeki metrik ölçümleri (Accuracy, Precision, Recall, F1-Score) aşağıdaki tabloda derlenmiştir. Bütün deneylerde Repulsive Hybrid stratejisiyle en iyi hiperparametreler saptanmıştır.

### 3.1. Nicel Sonuçlar (Karşılaştırma Tablosu)

| Model (Mimariler) | Optimizasyon | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **VGG16** | Repulsive Hybrid | **0.9289** | **0.9638** | **0.9020** | **0.8857** |
| **Custom CNN** | Repulsive Hybrid | 0.8147 | 0.7996 | 0.7722 | 0.7498 |
| **ResNet50** | Repulsive Hybrid | 0.6526 | 0.6139 | 0.6191 | 0.5350 |
| **Proposed CNN** | Repulsive Hybrid | 0.5670 | 0.5188 | 0.5931 | 0.4897 |

### 3.2. En İyi Modelin Hiperparametre Uzayı (VGG16)
VGG16 modelini %92.89 doğruluğa ulaştıran en ideal hiperparametre kombinasyonu TPE tarafından şu şekilde bulunmuştur:
- **Öğrenme Oranı (Learning Rate):** 8.569e-05
- **Dropout Oranı:** %33.0
- **Tam Bağlı Katman (Dense Units):** 384
- **Optimizasyon Algoritması:** RMSProp
- **Batch Size:** 32

### 3.3. Görsel Analizler (Plotlar)

> **Modellerin Genel Doğruluk ve F1 Karşılaştırması**  
> ![Final Comparison](results/plots/final_comparison.png)

> **En Başarılı Modelin (VGG16) Confusion Matrix ve Kayıp/Doğruluk Eğrileri**  
> ![VGG16 Confusion Matrix](results/plots/VGG16_BAYESIAN_cm.png)
> ![VGG16 Training History](results/plots/VGG16_BAYESIAN_history.png)

*(Not: Custom ve ResNet gibi diğer ağların matrix ve eğrileri `results/plots/` klasörü altında mevcuttur.)*

---

## 4. TARTIŞMA VE MİMARİ ANALİZ (Discussion)

Elde edilen deneysel sonuçlar üzerinden mimarilerin ve hiperparametre optimizasyon tekniklerinin analizi şu şekildedir:

1. **VGG16'nın Başarısı:** VGG16 (%92.89), daha sığ (daha az katmanlı) ve ardışık 3x3 konvolüsyonlardan oluşan yapısıyla endüstriyel doku hatalarını (texture defects) saptamada en iyi performansı göstermiştir. Resimlerdeki doku bozuklukları genel olarak düşük seviyeli (low-level) özniteliklerdir. VGG16'nın yapısı bu temel kenar ve doku farklılıklarını iyi yakalamış, ImageNet üzerinde önceden eğitilmiş ağırlıkları (pretrained weights) sayesinde kayıp fonksiyonunu (loss function) optimum karar sınırlarına (decision boundaries) kolayca ulaştırmıştır.
2. **ResNet50'nin Performans Kaybı:** ResNet50 (%65.26), VGG16'dan çok daha derin bir mimaridir ve residual (artık) bağlantılar kullanır. Ancak buradaki veri setinin doğası gereği problem "büyük objeleri tanımak" yerine "yüzeydeki çok ufak çatlak ve leke gibi tekrarlayan örüntüleri tanımak" olduğu için, modelin aşırı derinliği dezavantaj yaratmıştır. Ağ çok derinleştiğinde mikroskobik doku farklılıkları ileri yayılım sırasında kaybolmuş, model detaylara (fine-grained features) odaklanamamıştır.
3. **Sıfırdan Eğitilen Modeller (Custom & Proposed):** Custom CNN (%81.47) ve Proposed CNN (%56.70) modelleri sıfırdan (from scratch) eğitilmiştir. Sınıf başına sadece 50 eğitim örneği bulunması, bir CNN ağının filtre ağırlıklarını rastgele değerlerden başlayıp global bir minimuma indirmesi (yakınsaması) için yetersiz kalmıştır. Custom CNN içerdiği Squeeze-and-Excitation (SE) blokları sayesinde önemli öznitelik haritalarına (feature maps) dikkat ağırlığı vererek durumu nispeten toparlasa da, veri kısıtından dolayı önceden eğitilmiş (pretrained) bir modelin başarısına ulaşamamıştır.
4. **Repulsive Hybrid Yaklaşımı:** Hiperparametre optimizasyon sürecinde Genetik Algoritma'nın (GA) loss değerinin çok yüksek olduğu kötü parametre kümelerini tespit etmesi ve ardından Bayesyen (TPE) optimizasyonun arama uzayını bu kısımlardan uzaklaştırması (Repulsive), modelin az sayıda denemede (trial) hızlıca minimuma yakınsamasını (convergence) sağlamıştır.

**Sonuç:** Kısıtlı veriye sahip endüstriyel hata tespit problemlerinde, çok derin olmayan sıralı ağların (VGG16) transfer learning ile kullanılması ve hiperparametre arama uzayının akıllı yöntemlerle (Hybrid GA+TPE) optimize edilmesi en güvenilir sonucu vermektedir.

---

## 5. EKLER (Appendix)

### 5.1. Veri Seti Kaynağı
- **Orijinal Akademik Kaynak:** Üniversitenin resmi akademik web sitesinden temin edilmiştir. (Kaggle veya harici bir aracı platform kullanılmamıştır.)

### 5.2. Kullanılan Kütüphaneler ve Amaçları
- **PyTorch & Torchvision (`torch`, `torchvision`):** Veri setinin tensor dönüşümleri, transfer learning (VGG, ResNet) modellerinin import edilmesi, loss ve gradient hesaplamalarının GPU (CUDA 12.4) üzerinde matrisel olarak işletilmesi için.
- **Optuna (`optuna`):** İstatistiksel hiperparametre optimizasyonu olan Bayesyen TPE yönteminin dinamik arama uzayını inşa etmek ve budama (pruning) yapmak için.
- **DEAP (`deap`):** Genetik Algoritma'nın kromozom kodlamasını (1B gen dizilimini), rulet/turnuva seçilimini ve iki noktalı çaprazlama işlemlerini simüle etmek için.
- **Scikit-Learn (`sklearn`):** Model doğrulandıktan sonra matematiksel metriklerin (Macro Precision, Recall, F1-Score) ve Confusion Matrix'in IMRAD standardında hatasız hesaplanması için.
- **Matplotlib & Seaborn (`matplotlib`, `seaborn`):** Eğim (Loss) ve başarım (Accuracy) metriklerinin epoch'lara bağlı geometrik değişimlerini, matrix ısı haritalarını görsel olarak ifade edebilmek için.

### 5.3. Referans
- Acici, K., Beyaz, S., Sumer, E. (2020). *Femoral neck fracture detection in X-ray images using deep learning and genetic algorithm approaches.* Joint Diseases and Related Surgery, 31(2), 175-183.
