# Harmonia v2 — CBIS-DDSM Meme Kanseri Segmentasyonu

**ConnectedUnet + EfficientNet-B3 ile ROI Odaklı Tümör Segmentasyonu | PyTorch**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dice](https://img.shields.io/badge/Test%20Dice-0.917-brightgreen.svg)](#sonuçlar)
[![IoU](https://img.shields.io/badge/Test%20IoU-0.852-brightgreen.svg)](#sonuçlar)

---

## Genel Bakış

Bu proje, CBIS-DDSM mammografi veri seti üzerinde meme kanseri tümör segmentasyonu gerçekleştiren uçtan uca bir derin öğrenme pipelinedır. Sıfırdan eğitilmiş **EfficientNet-B3 encoder** ve **ConnectedUnet decoder** kullanan bu sistem, 256×256 gri tonlamalı ROI kesimlerinden ikili tümör maskesi üretmektedir.

Proje iki ana bileşenden oluşmaktadır:

1. **Veri Hazırlama Pipeline** (`scripts/`) — CBIS-DDSM veri temizleme ve ön işleme (11 adım)
2. **Segmentasyon Modeli** (`segmentation/`) — ConnectedUnet eğitimi, değerlendirmesi ve FL desteği
3. **Federated Learning Benchmark** (`fl/`, `benchmark/`) — Flower FedAvg, IID/Non-IID hasta bölümleri ve TenSEAL CKKS PHE deneyleri

---

## Sonuçlar

Test seti üzerinde (260 örnek, hasta bazlı bölme):

| Metrik | Ortalama | Std | Medyan |
|---|---|---|---|
| **Dice** | **0.9174** | ±0.0565 | 0.9297 |
| **IoU** | **0.8518** | ±0.0843 | — |
| Sensitivity (Recall) | 0.9397 | ±0.0535 | — |
| Specificity | 0.9281 | ±0.0581 | — |
| Precision | 0.9048 | ±0.0910 | — |
| Pixel Accuracy | 0.9292 | — | — |

Tüm örnek bazlı sonuçlar: [`outputs/segmentation/evaluation/evaluation_results.json`](outputs/segmentation/evaluation/evaluation_results.json)

Eğitim geçmişi (70 epoch): [`outputs/segmentation/training_history.json`](outputs/segmentation/training_history.json)

### Federated Learning Benchmark

FL fazı için C1/F1/F2/F3/F4 karşılaştırmalı rapor:
[`outputs/benchmark_report/report.md`](outputs/benchmark_report/report.md)

Makine-okunur özet tablo:
[`outputs/benchmark_report/comparison_table.csv`](outputs/benchmark_report/comparison_table.csv)

Rapor; final test Dice/IoU/Sensitivity/Specificity/Precision, piksel karmaşıklık sayımları, en iyi/final validation metrikleri, round sayısı, süre ve upload/download iletişim metriklerini içerir.

---

## Klasör Yapısı

```
harmonia_v2/
├── scripts/                              # Veri ön işleme (adımlar 1–11)
│   ├── config.py                         # Paylaşılan yollar ve parametreler
│   ├── step1_load_and_merge.py           # CSV yükleme ve birleştirme
│   ├── step2_join_metadata.py            # DICOM metadata eşleştirme
│   ├── step3_consolidate_labels.py       # Etiket konsolidasyonu
│   ├── step4_file_integrity.py           # Dosya bütünlüğü kontrolü
│   ├── step5_mask_validation.py          # İkili maske doğrulama
│   ├── step6_combine_masks.py            # Çoklu maske çözümleme
│   ├── step7_crop_mask_validation.py     # Uzamsal hizalama doğrulama
│   ├── step8_crop_resize.py              # Pad + 256×256 yeniden boyutlandırma
│   ├── step9_split_data.py               # Hasta bazlı 70/15/15 bölme
│   ├── step10_save_npy.py                # .npy dizi aktarımı
│   └── step11_create_fl_partitions.py    # FL istemci bölümleri (IID/Non-IID)
├── segmentation/                         # Segmentasyon modeli
│   ├── config.py                         # Tüm hiperparametreler (tek kaynak)
│   ├── model/
│   │   ├── efficientnet.py               # EfficientNet-B3 encoder (sıfırdan)
│   │   └── connected_unet.py             # ConnectedUnet + attention gate decoder
│   ├── data/
│   │   ├── dataset.py                    # CBISDDSMDataset loader
│   │   └── transforms.py                 # Augmentation (elastic, affine, flip)
│   ├── training/
│   │   ├── trainer.py                    # Eğitim döngüsü (AMP, GradScaler)
│   │   ├── losses.py                     # CombinedLoss (BCE+Dice+Focal+Tversky)
│   │   └── metrics.py                    # Dice, IoU, Sensitivity, Specificity
│   └── evaluation/
│       └── evaluator.py                  # Test seti değerlendirme
├── run.py                                # Ana giriş noktası (train / evaluate)
├── outputs/
│   ├── final_npy/                        # Model-ready .npy dosyaları (git-ignored)
│   │   └── pipeline_summary.json         # Pipeline istatistikleri
│   └── segmentation/
│       ├── config.json                   # Kayıtlı eğitim konfigürasyonu
│       ├── training_history.json         # 70 epoch eğitim geçmişi
│       └── evaluation/
│           └── evaluation_results.json   # Test seti metrikleri
├── data_cleaning_steps.md                # Pipeline adım adım dokümantasyonu
├── requirements.txt
└── README.md
```

---

## Kurulum

### 1. Klonla ve ortam oluştur

```bash
git clone <repo-url>
cd harmonia_v2
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

> **Not:** PyTorch CUDA sürümünü NVIDIA sürücünüze göre [pytorch.org](https://pytorch.org/get-started/locally/) adresinden seçerek ayrıca yükleyin.

### 2. Veri setini indir

[Kaggle — CBIS-DDSM (awsaf49)](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/data) adresinden veri setini indir. `csv/` ve `jpeg/` klasörlerini aşağıdaki yapıya uygun şekilde çıkar:

```
harmonia_v2/
└── CBIS-DDSM Dataset/
    ├── csv/
    │   ├── mass_case_description_train_set.csv
    │   ├── mass_case_description_test_set.csv
    │   ├── dicom_info.csv
    │   └── ...
    └── jpeg/
        └── <UID klasörleri>/
```

### 3. Yol yapılandırması

`scripts/config.py` ve `segmentation/config.py` içindeki `ROOT` değişkenini kendi proje dizinine güncelle:

```python
# scripts/config.py
ROOT = Path(r"C:\kullanici\projeler\harmonia_v2")

# segmentation/config.py
ROOT = Path(r"C:\kullanici\projeler\harmonia_v2")
```

---

## Kullanım

### Adım 1: Veri ön işleme pipeline

```bash
cd scripts
python step1_load_and_merge.py
python step2_join_metadata.py
python step3_consolidate_labels.py
python step4_file_integrity.py
python step5_mask_validation.py
python step6_combine_masks.py
python step7_crop_mask_validation.py
python step8_crop_resize.py
python step9_split_data.py
python step10_save_npy.py
python step11_create_fl_partitions.py
```

Her adım idempotent — bir önceki adımın çıktısını okur, kendi çıktısını üretir. Yeniden çalıştırmak güvenlidir.

Pipeline çıktısı `outputs/final_npy/` dizinine kaydedilir ve segmentasyon modeli buradan okur.

### Adım 2: Segmentasyon modeli

Tüm komutlar proje kök dizininden (`harmonia_v2/`) çalıştırılır.

**Yeni model eğitimi:**
```bash
python run.py train
```

**Test seti değerlendirmesi:**
```bash
python run.py evaluate
```

**Eğitim + değerlendirme (tek komut):**
```bash
python run.py train-eval
```

**Checkpoint'ten devam:**
```bash
python run.py train --resume outputs/segmentation/checkpoints/best_model.pt
```

**Belirli checkpoint ile değerlendirme:**
```bash
python run.py evaluate --ckpt outputs/segmentation/checkpoints/best_model.pt
```

---

## Segmentasyon Modeli

### Mimari

| Bileşen | Detay |
|---|---|
| Encoder | EfficientNet-B3 (width=1.2, depth=1.4, sıfırdan eğitilmiş) |
| Decoder | ConnectedUnet (5 yukarı örnekleme katmanı) |
| Decoder kanalları | 320 → 160 → 80 → 40 → 20 |
| Skip connection | Attention gate |
| Giriş | 256×256×1 (gri tonlamalı) |
| Çıkış | 256×256×1 (sigmoid ikili maske) |
| Mixed Precision | AMP float16 (torch.amp) |

### Kayıp Fonksiyonu (CombinedLoss)

~%43 tümör kapsama oranına (neredeyse dengeli) göre kalibre edilmiştir:

| Kayıp | Ağırlık | Ayarlar |
|---|---|---|
| BCE | 0.3 | pos_weight=1.5 |
| Dice | 0.5 | smooth=1e-4 (float16 güvenli) |
| Focal | 0.1 | α=0.5, γ=2.0 |
| Tversky | 0.1 | α=0.4, β=0.6 |

### Eğitim Yapılandırması

| Parametre | Değer |
|---|---|
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-5) |
| Scheduler | Cosine annealing + 10 epoch warmup |
| Toplam epoch | 100 (early stopping patience=25) |
| Batch size | 8 |
| Gradient clipping | max_norm=1.0 |
| Augmentation | Elastic, affine, horizontal flip, brightness/contrast |

Konfigürasyonun tamamı için: [`segmentation/config.py`](segmentation/config.py)

---

## Veri Seti

**CBIS-DDSM** (Curated Breast Imaging Subset — Digital Database for Screening Mammography)

- Kaynak: [Kaggle awsaf49](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)

| Bölüm | Örnek | Hasta | Benign | Malignant |
|---|---|---|---|---|
| Train | 1132 | 613 | 603 | 529 |
| Validation | 238 | 132 | 117 | 121 |
| Test | 260 | 132 | 154 | 106 |
| **Toplam** | **1630** | **877** | **874** | **756** |

- Görüntü: 256×256 gri tonlamalı (uint8)
- Maske: 256×256 ikili (0/255 uint8, eşik=127.5)
- Tümör kapsama oranı: ~%43 ortalama (neredeyse dengeli)
- Bölme stratejisi: **Hasta bazlı** — aynı hasta birden fazla bölümde yer almaz (veri sızıntısı önlenir)

---

## Önemli Teknik Notlar

### CBIS-DDSM Maske Hizalama Sorunu

CBIS-DDSM'in kritik bir veri yapısı özelliği: ROI maske dosyaları **tam mamogram çözünürlüğünde** (~3000-5500 px) depolanır, ancak eşleşen crop görselleri ~200-500 px sıkı ROI patchleridir.

Maskeyi doğrudan 256×256'ya yeniden boyutlandırmak, tümörü yalnızca %0.44 kaplayan bir noktaya indirgiyordu; oysa crop görselinde tümör çerçevenin %60-80'ini kaplıyordu.

**Çözüm** (`scripts/step8_crop_resize.py` — `extract_mask_crop()`): Tam çözünürlüklü maskeden tümör bbox merkezine hizalanmış bir pencere çıkarılarak crop görseliyle aynı uzamsal bölge elde edilir.

**Sonuç:** Tümör kapsama oranı %0.44 → ~%43 (neredeyse dengeli sınıf dağılımı).

### Float16 AMP ve Dice Smooth

Float16'nın minimum normal değeri ~6.1×10⁻⁵'tir. `dice_smooth=1e-7` değeri underflow'a yol açarak Dice gradyanlarını bozar. Güvenli değer: **`dice_smooth=1e-4`**.

---

## Katkıda Bulunanlar

- **Berk Günberk**
- **Eyüpcan Işıkgör**
- **Beril Çitil**
- **İrem Batıgün**
- **Kutay Orallı**
