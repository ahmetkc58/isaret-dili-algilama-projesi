# Sessiz Kelimeler (Silent Words)

Türk işaret dilindeki kelimeleri tanıyan, derin öğrenme ve Capsule Network mimarisi kullanarak geliştirilen yapay zeka projesi.

## Proje Açıklaması

Bu proje, **TÜBİTAK** desteğiyle geliştirilen işaret dili tanıma sistemidir. MobileNet ve Capsule Network mimarileri kullanarak kamera üzerinden çekilen videolar üzerinden işaret dili kelimelerini gerçek zamanlı olarak tanır ve tahmin eder.

### Temel Özellikler

- Gerçek zamanlı kamera üzerinden işaret dili tanıma
- MobileNet + Capsule Network mimarisi
- FastAPI ile REST API sunucusu
- Confusion matrix ve model performans analizi
- Ses entegrasyonu (pyttsx3)
- Akıllı kelime tahmini ve fuzzy matching

## Teknolojiler

### Framework & Kütüphaneler
- **TensorFlow/Keras** - Derin öğrenme modeli
- **FastAPI** - REST API sunucusu
- **OpenCV** - Görüntü işleme
- **NumPy & Pandas** - Veri işleme
- **scikit-learn** - Model değerlendirme ve fuzzy matching
- **pyttsx3** - Metin-konuşma dönüştürme

### Sistem Gereksinimleri
- Python 3.8+
- CUDA & cuDNN (GPU desteği için)
- Webcam/Kamera

## Veri Seti

Bu proje, Kaggle'dan alınan Türkçe İşaret Dili veri setini kullanmaktadır.

**Veri Seti Kaynağı:**
- [TR Sign Language Dataset - Kaggle](https://www.kaggle.com/datasets/berkaykocaoglu/tr-sign-language)

Veri setini indirmek için:
1. Kaggle hesabınıza giriş yapın
2. Yukarıdaki linki ziyaret edin
3. "Download" butonuna tıklayın
4. İndirilen dosyaları `datasets/` klasörüne yerleştirin

## Kurulum

### 1. Repository'i Clone Edin
```bash
git clone https://github.com/yourusername/SessizKelimeler.git
cd SessizKelimeler
```

### 2. Virtual Environment Oluşturun
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Gerekli Paketleri Yükleyin
```bash
pip install -r requirements.txt
```

## Kullanım

### API Sunucusunu Başlatın
```bash
python api.py
```
Sunucu `http://localhost:8000` adresinde başlar.

**API Endpoints:**
- `POST /predict` - Resim yükleme ve tahmin
- `GET /docs` - Swagger API dokümantasyonu

### Kamera Testi
```bash
python kamera_test.py
```
Gerçek zamanlı işaret dili tanıma için webcam kullanır.

### Ses Testi
```bash
python ses_test.py
```
Tanınan kelimeleri sesli olarak okur.

### Video Dashboard (Raspberry Pi)
Elinizdeki videoyu izlerken aynı anda model tahminlerinden cümle üretmek için:

```bash
# Windows PowerShell örneği
$env:VIDEO_PATH="C:\\path\\to\\video.mp4"
python video_dashboard.py
```

Dashboard adresi:
- `http://localhost:8010` (video + canlı cümle)
- `http://localhost:8010/state` (JSON durum)

Opsiyonel ayarlar (ortam değişkenleri):
- `MODEL_PATH` (varsayılan: `best_model.h5`)
- `WORDLIST_PATH` (varsayılan: `turizm_sozluk.json`)
- `CONFIDENCE_THRESHOLD` (varsayılan: `0.60`)
- `STABLE_REQUIRED` (varsayılan: `4`)
- `PROCESS_EVERY_N_FRAME` (varsayılan: `2`)

## Proje Yapısı

```
SessizKelimeler/
├── api.py                           # FastAPI sunucusu
├── mobilenet_capsule_network.py    # Model mimarisi
├── kamera_test.py                  # Kamera üzerinden test
├── ses_test.py                     # Ses testi
├── best_model.h5                   # Eğitilmiş model
├── turizm_sozluk.json              # Kelime sözlüğü
├── datasets/                       # Eğitim veri seti
├── sessizkelimeler.ipynb           # Jupyter notebook (Eğitim)
├── requirements.txt                # Paket bağımlılıkları
└── README.md                       # Bu dosya
```

## Model Mimarisi

Proje, **MobileNet** ve **Capsule Network** kombinasyonu kullanır:

1. **MobileNet**: Hızlı ve hafif öznitelik çıkarma
2. **Capsule Network**: Hiyerarşik yapı ve konumsal bilgi koruması
3. **Primary Capsule Layer**: Temel kapsül oluşturma
4. **Routing Algorithm**: Kapsüller arası iletişim

## Model Performansı

- Eğitim doğruluğu: ~95%+
- Gerçek zamanlı tahmin hızı: 30+ FPS
- Fuzzy matching ile kelime benzerliği: %90+

## Katkılar

Bu proje TÜBİTAK tarafından desteklenmiş olan resmi bir projedir.

## Lisans

Bu proje [TÜBİTAK KIRKEN 2209-A] programı kapsamında geliştirilmiştir.

## Kaynaklar

- [Capsule Networks - Geoffrey Hinton](https://arxiv.org/abs/1710.09829)
- [MobileNets: Efficient Convolutional Neural Networks](https://arxiv.org/abs/1704.04861)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

Bu projeyi faydalı buldum ise yıldız vermeyi unutmayın!
