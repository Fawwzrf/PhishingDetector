# 🎣 Phishing Detector (Machine Learning Pipeline)

[![Python Support](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.2%2B-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-blue.svg)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-3.5%2B-blueviolet.svg)](https://optuna.org/)

Repositori ini memuat *End-to-End Machine Learning Pipeline* untuk mendeteksi *website phishing* (situs penipuan) berdasarkan ekstraksi fitur-fitur dari URL dan konten web. Proyek ini dibangun dengan arsitektur kode tingkat produksi (*production-grade*) menggunakan pustaka kustom `mltools`.

---

## 🚀 Fitur Utama

- **Pipeline Preprocessing Terstruktur:** Penanganan nilai yang hilang secara *expert* (*IterativeImputer*, KNN), deteksi *outlier* tingkat lanjut (*IsolationForest*, *LocalOutlierFactor*), serta *feature engineering* kustom.
- **Auto EDA (Exploratory Data Analysis):** Integrasi pembuatan laporan otomatis untuk analisis statistik dan distribusi data (*Sweetviz* / *YData Profiling*).
- **Hyperparameter Tuning Lanjutan:** Menggunakan **Optuna** dengan *Tree-structured Parzen Estimator* (TPE) *sampler* untuk proses *tuning* yang sangat optimal pada model *Tree-based*.
- **Support Multi-Model Ensemble:** Mendukung *training* komparatif lintas model seperti **LightGBM**, **XGBoost**, **CatBoost**, dan **Random Forest**.
- **Konfigurasi Berbasis YAML:** Semua *hyperparameter*, jalur direktori (*paths*), dan konfigurasi model disentralisasi menggunakan YAML dan Pydantic, sehingga kode tidak perlu diubah (*zero-code change*) untuk eksperimen baru.
- **Model Explainability:** Dilengkapi visualisasi **SHAP** untuk interpretasi fitur dominan.

## 📂 Struktur Direktori

```text
PhishingDetector/
├── configs/                 # Konfigurasi pipeline (ml_config.yaml)
├── data/
│   ├── raw/                 # Dataset awal (mentah)
│   ├── interim/             # Data sementara hasil proses parsial
│   └── processed/           # Data bersih siap latih (Train/Test Split)
├── models/                  # Direktori penyimpanan model (.pkl, .joblib)
├── notebooks/               # Eksperimen Jupyter Nodebook / Colab
├── reports/                 # Hasil metrik, grafik (SHAP, ROC), dan Log
├── src/
│   └── mltools/             # (Core Library) Pustaka custom pipeline ML
├── README.md                # Dokumentasi Proyek Phishing Detector (File ini)
├── README_MLTOOLS.md        # Dokumentasi internal framework mltools
├── requirements.txt         # Daftar dependensi & versi library
└── pyproject.toml           # Konfigurasi proyek & build-system python
```

## 🛠️ Instalasi & Persiapan

1. **Clone Repositori:**
   ```bash
   git clone https://github.com/Fawwzrf/PhishingDetector.git
   cd PhishingDetector
   ```

2. **Buat Virtual Environment (Opsional namun disarankan):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   # atau
   # venv\Scripts\activate   # Untuk Windows
   ```

3. **Instal Dependensi & Backend:**
   *Repository* ini dirancang agar `src/mltools` dapat dikenali sebagai sebuah modul (Pustaka). Jalan perintah berikut:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## 🧠 Proses Eksperimen (Google Colab / Jupyter)

Proyek ini telah dibagi menjadi dua tahap eksekusi besar yang biasanya dijalankan pada *notebook* untuk fleksibilitas:

1. **Tahap Preprocessing (`01_preprocessing.ipynb`)**
   - Membersihkan data fitur URL.
   - Mengatasi multikolinearitas (menghapus fitur yang *redundant*).
   - Menghasilkan output di `data/processed/` (misal: `X_train.csv`, `X_test.csv`, dll).

2. **Tahap Modelling & Tuning (`02_modelling.ipynb`)**
   - Melatih model dengan data yang sudah bersih.
   - Optuna Tuning untuk mencari *Best Hyperparameters* (Contoh: `lgbm`, `xgb`).
   - Evaluasi menggunakan *ROC-AUC*, *F1-Score*, dan visualisasi iterasi.
   - Hasil akhir model tersimpan di direktori `models/`.

## ⚙️ Cara Menyesuaikan Konfigurasi

Anda dapat mengubah *behavior* Pipeline hanya dengan mengedit `configs/ml_config.yaml`.
Contoh:
```yaml
modeling:
  metric: "roc_auc"
  n_cv_folds: 5
  tuning:
    n_trials: 50      # Ubah jumlah iterasi Optuna di sini
    timeout: 3600     # Maksimal waktu tuning (detik)
```

## 📝 Catatan Library Internal (`mltools`)
Jika Anda ingin melihat cara pustaka `mltools` bekerja secara spesifik untuk membangun *Pipeline* ini, silakan merujuk ke [README_MLTOOLS.md](README_MLTOOLS.md).

---
*Dikembangkan untuk eksperimen deteksi Phishing tingkat lanjut.*
