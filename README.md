# DAGM 2007 — CNN Hiperparametre Optimizasyonu
## PyTorch + RTX 4060 | GA vs Bayesian TPE | 10 Sınıf

### Klasör Yapısı
```
dagm_pt/
├── data/
│   └── DAGM2007/
│       ├── raw/              ← ZIP'leri buraya aç
│       │   ├── Class1/
│       │   │   ├── Train/    ← PNG görüntüler (normal + defect)
│       │   │   └── Test/
│       │   ├── Class2/ ... Class10/
│       ├── train/            ← prepare_dataset.py oluşturur
│       ├── val/
│       └── test/
├── models/
│   ├── vgg16_model.py
│   ├── resnet50_model.py
│   └── custom_cnn_model.py
├── optimization/
│   ├── genetic_algorithm.py
│   └── bayesian_tpe.py
├── utils/
│   ├── data_loader.py
│   ├── trainer.py
│   ├── metrics.py
│   └── plotter.py
├── results/
│   ├── ga/
│   ├── bayesian/
│   ├── plots/
│   └── models/
├── prepare_dataset.py
├── train_and_compare.py
└── requirements.txt
```

### Kurulum
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install optuna scikit-learn matplotlib seaborn Pillow tqdm deap pandas
```

### Çalıştırma
```bash
python prepare_dataset.py
python train_and_compare.py
```

### Dataset
- Kaynak: https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection
- 10 sınıf (farklı endüstriyel doku tipleri)
- ~11500 görüntü (sınıf başına ~1150)
- Format: Grayscale 8-bit PNG, 512×512
- Referans: Wieler & Hahn, DAGM Workshop 2007
