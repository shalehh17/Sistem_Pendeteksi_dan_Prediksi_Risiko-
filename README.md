# 🚚 Logistics AI Concierge: Intelligent Route & Delivery Delay Risk Prediction System

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%2FKeras-orange.svg)](https://tensorflow.org/)
[![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-%3E%2097.7%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 📌 1. Deskripsi Proyek

**Logistics AI Concierge** adalah platform cerdas berbasis Machine Learning dan Deep Learning (*Multilayer Perceptron Neural Network*) yang dirancang untuk memprediksi tingkat risiko keterlambatan pengiriman paket (*delivery delay risk*) secara *real-time* serta menyediakan rekomendasi mitigasi operasional yang proaktif bagi industri logistik & rantai pasok (*supply chain*).

Sistem ini menganalisis berbagai variabel fisik dan kontekstual pengiriman (seperti jarak, estimasi SLA, cuaca, mitra ekspedisi, jenis armada, dan beban paket) untuk mengklasifikasikan status akhir paket menjadi:
1. **Delivered** (Tepat Waktu / Sukses)
2. **Delayed** (Mengalami Keterlambatan)
3. **Failed** (Gagal Terkirim)

Selain modul inferensi prediksi, sistem ini dilengkapi dengan **Live-Feed Interactive Analytics Dashboard** yang menghubungkan data historis dan setiap input prediksi baru dari pengguna secara langsung tanpa perlu melakukan proses *retrain* model.



---

## 🏗️ 2. Arsitektur & Struktur Proyek

Proyek ini dibangun menggunakan arsitektur modular *production-ready* yang memisahkan antara pipeline data, pelatihan model, dan modul antarmuka pengguna (*views*).

### 📁 Struktur Direktori
```text
Sistem_Pendeteksi_dan_Prediksi_Risiko-/
│
├── app/
│   ├── __init__.py                # Inisialisasi package modular app
│   ├── main.py                    # Entry point utama aplikasi Streamlit (Router)
│   └── views/
│       ├── __init__.py            # Inisialisasi package views
│       ├── prediction.py          # Modul inferensi form input & integrasi state
│       └── dashboard.py           # Modul dasbor analitik real-time & filtering
│
├── src/
│   ├── __init__.py                # Inisialisasi source package
│   ├── utils.py                   # Path config & helper functions
│   ├── preprocessing.py           # Data pipeline & transformer builder (.pkl)
│   └── train.py                   # Deep Learning MLP training pipeline (.h5)
│
├── data/
│   └── logistics_data.csv         # Dataset historis pengiriman multi-partner
│
├── models/
│   ├── preprocessor.pkl           # Artefak pipeline scikit-learn
│   └── dl_model.h5                # Artefak model Deep Learning TensorFlow


## 🧠 Arsitektur Machine Learning Pipeline

Sistem ini mengimplementasikan *end-to-end data pipeline* mulai dari penyerapan data mentah operasional, pra-pemrosesan terstandarisasi, pemodelan *Deep Learning & Machine Learning*, hingga tahap inferensi *real-time* berbasis antarmuka Streamlit.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           1. DATA INGESTION & SOURCING                      │
│   - Multi-partner Logistics Dataset (Jarak, Berat, Cuaca, Kurir, Durasi)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           2. DATA PREPROCESSING & FEATURE PIPELINE          │
│   ├── Missing Values & Outlier Handling                                     │
│   ├── Feature Engineering (Speed, Cost Ratio, Weather-Traffic Risk Factor)  │
│   ├── Categorical Encoding (One-Hot / Target Encoder for Partners & Routes) │
│   └── Feature Scaling (StandardScaler / MinMaxScaler -> artifacts/*.joblib) │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           3. MODELING & EXPERIMENTATION                     │
│   ├── Deep Neural Network (DNN / MLP) via TensorFlow/Keras                  │
│   ├── Gradient Boosting / Random Forest (Scikit-Learn)                      │
│   └── Model Optimization (Hyperparameter Tuning, Cross-Validation)          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           4. EVALUATION & SERIALIZATION                     │
│   ├── Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix      │
│   └── Artifact Export: `model.h5` / `model.joblib` & `preprocessor.joblib`  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           5. DEPLOYMENT & INFERENCE ENGINE                  │
│   ├── Real-time Risk Prediction (Status: On-Time / Delayed / Failed)        │
│   ├── Dynamic Mitigation & Business Recommendations                         │
│   └── Interactive Streamlit UI & Live Analytics Dashboard                   │
└─────────────────────────────────────────────────────────────────────────────┘



