# 🚚 Logistics AI Concierge: Intelligent Route & Delivery Delay Risk Prediction System

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%2FKeras-orange.svg)](https://tensorflow.org/)
[![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-%3E%2097.7%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 📌 1. Deskripsi Proyek

**Logistics AI Concierge** adalah platform cerdas berbasis *Machine Learning* dan *Deep Learning* (*Multilayer Perceptron Neural Network*) yang dirancang untuk memprediksi tingkat risiko keterlambatan pengiriman paket (*delivery delay risk*) secara *real-time* serta menyediakan rekomendasi mitigasi operasional yang proaktif bagi industri logistik & rantai pasok (*supply chain*).

Sistem ini menganalisis berbagai variabel fisik dan kontekstual pengiriman (seperti jarak, estimasi SLA, cuaca, mitra ekspedisi, jenis armada, dan beban paket) untuk mengklasifikasikan status akhir paket menjadi:
1. **Delivered** (Tepat Waktu / Sukses)
2. **Delayed** (Mengalami Keterlambatan)
3. **Failed** (Gagal Terkirim)

Selain modul inferensi prediksi, sistem ini dilengkapi dengan **Live-Feed Interactive Analytics Dashboard** yang menghubungkan data historis dan setiap input prediksi baru dari pengguna secara langsung tanpa perlu melakukan proses *retrain* model.

---

## 📁 2. Struktur Direktori & Project

Proyek ini dibangun menggunakan arsitektur modular *production-ready* yang memisahkan antara pipeline pemrosesan data, pelatihan model, serta modul antarmuka pengguna (*views*):

```text
Sistem_Pendeteksi_dan_Prediksi_Risiko-/
│
├── 📂 app/                                # Frontend & Application Layer
│   ├── __init__.py                        # Inisialisasi package modular app
│   ├── main.py                            # Entry point utama aplikasi Streamlit (Router)
│   └── 📂 views/                          # Modular view pages
│       ├── __init__.py                    # Inisialisasi package views
│       ├── prediction.py                  # Modul inferensi form input & integrasi state
│       └── dashboard.py                   # Modul dasbor analitik real-time & filtering
│
├── 📂 src/                                # Core Source Code & Pipeline Logic
│   ├── __init__.py                        # Inisialisasi source package
│   ├── utils.py                           # Path config & helper functions
│   ├── preprocessing.py                   # Data pipeline & transformer builder (.pkl)
│   └── train.py                           # Deep Learning MLP training pipeline (.h5)
│
├── 📂 data/                               # Data Store Layer
│   └── logistics_data.csv                 # Dataset historis pengiriman multi-partner
│
├── 📂 models/                             # Model Artifacts Storage
│   ├── preprocessor.pkl                   # Artefak pipeline scikit-learn (Scaler & Encoder)
│   └── dl_model.h5                        # Artefak model Deep Learning TensorFlow/Keras
│
├── 📄 .gitignore                          # Filter file cache, environment, & binary models
├── 📄 requirements.txt                    # Daftar pustaka dependencies Python
└── 📄 README.md                           # Dokumentasi komprehensif proyek
```

## 🧠 3. Arsitektur Machine Learning Pipeline
Pipeline data dan pemodelan dirancang secara berurutan mulai dari konsumsi dataset logistik multi-partner hingga inferensi prediktif:

🔄 Alur Pipeline End-to-End
┌─────────────────────────────────────────────────────────────────────────────┐
│                           1. DATA INGESTION & SOURCING                      │
│   - Multi-partner Logistics Dataset (Jarak, Berat, Cuaca, Kurir, Durasi)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           2. FEATURE ENGINEERING & TRANSFORMATION           │
│   ├── Numerical Features (Robust / StandardScaler):                         │
│   │   └── distance_km, package_weight_kg, delivery_time_hours, dll.         │
│   ├── Categorical Features (One-Hot Encoding):                              │
│   │   └── delivery_partner, region, weather_condition, vehicle_type,        │
│   │       package_type, delivery_mode                                       │
│   └── Serialization: Export pipeline ke models/preprocessor.pkl             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           3. DEEP LEARNING ARCHITECTURE (MLP)               │
│   ├── Type: Multilayer Perceptron (MLP) via TensorFlow / Keras              │
│   ├── Hidden Layers: Dense Layers + ReLU + Batch Normalization              │
│   ├── Regularization: Dropout Layers (rate: 0.2 - 0.3)                      │
│   ├── Optimization: Adam Optimizer & Categorical Crossentropy Loss          │
│   └── Output Layer: Softmax Multi-class Classifier                          │
│       ├── Delivered (Tepat Waktu)                                           │
│       ├── Delayed   (Terlambat)                                             │
│       └── Failed    (Gagal)                                                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           4. EVALUATION & ARTIFACT EXPORT                   │
│   ├── Validation Metrics: Model Accuracy (> 97.7%), Precision, Recall, F1   │
│   └── Artifact Export: Bobot model diekspor ke models/dl_model.h5           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           5. REAL-TIME SERVING & MITIGATION ENGINE          │
│   ├── Form Input Interaktif (app/views/prediction.py)                       │
│   ├── Threshold Scoring: <40% Optimal | 40-70% Sedang | >70% Risiko Tinggi │
│   ├── Dynamic Business Decision Engine (Mitigasi Proaktif)                  │
│   └── Live-Feed State Management & Auto Dashboard Sync                      │
└─────────────────────────────────────────────────────────────────────────────┘

