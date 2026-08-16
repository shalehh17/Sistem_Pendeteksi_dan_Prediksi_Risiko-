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

<img width="1505" height="8192" alt="Christmas Shopping Decision-2026-08-16-100317" src="https://github.com/user-attachments/assets/dcf6ce79-02b5-4ee7-b5a3-d618cfd74bdd" />



### 🔬 Komponen & Spesifikasi Pipeline

* **Feature Engineering & Transformation:**
  * **Numerical Features (Robust Scaling / Standardization):** `distance_km`, `package_weight_kg`, `delivery_time_hours`, dll.
  * **Categorical Features (One-Hot Encoding):** `delivery_partner`, `region`, `weather_condition`, `vehicle_type`, `package_type`, `delivery_mode`.
* **Model Architecture:**
  * **Type:** Deep Neural Network (*Multilayer Perceptron / MLP*).
  * **Layers:** Dense Layers dengan ReLU Activation, Batch Normalization, dan Dropout layers (0.2–0.3) untuk regularisasi.
  * **Output Layer:** Softmax Multi-class Classifier (`Delivered`, `Delayed`, `Failed`).
  * **Optimization:** Adam Optimizer, Categorical Crossentropy Loss.


---

## ✨ 4. Fitur Utama

* 🔍 **Real-Time Risk Prediction**: Form interaktif untuk memasukkan parameter operasional dan menghitung probabilitas delay dalam hitungan milidetik.
* 🚦 **Proactive Decision Engine**: Aturan mitigasi bisnis dinamis berbasis threshold risiko (Risiko Tinggi >70%, Sedang 40–70%, Optimal <40%).
* 📊 **Dynamic Analytics Dashboard**: Visualisasi distribusi historis dengan kemampuan filter interaktif (Mitra, Wilayah, Cuaca).
* ⚡ **Live-Feed Reactive Data Flow**: Setiap transaksi baru yang dihitung di form prediksi akan otomatis masuk ke memori sesi (`st.session_state`) dan memperbarui KPI serta grafik analitik secara instan.


---

## 🚀 5. Langkah-Langkah Menjalankan Proyek (Step-by-Step)

Ikuti langkah-langkah berikut untuk meng-clone, menyiapkan environment, dan menjalankan aplikasi di komputer lokal Anda:

### Langkah 1: Clone Repositori
Buka terminal Anda dan jalankan perintah berikut:
```bash
git clone [https://github.com/shalehh17/Sistem_Pendeteksi_dan_Prediksi_Risiko-.git](https://github.com/shalehh17/Sistem_Pendeteksi_dan_Prediksi_Risiko-.git)
cd Sistem_Pendeteksi_dan_Prediksi_Risiko-


---
## Langkah 2: Buat Virtual Environment

# Untuk Windows (Command Prompt / PowerShell)
python -m venv venv
venv\Scripts\activate

# Untuk macOS / Linux
python3 -m venv venv
source venv/bin/activate

----
## Langkah 3: Install Dependensi
Bash
pip install -r requirements.txt
Langkah 4: Bangun Artefak Model (Jika Belum Tersedia)
Jalankan modul preprocessing data dan training model secara modular:

------

## Bash
# 1. Bangun pipeline preprocessor
python -m src.preprocessing

---
# 2. Latih model Deep Learning
python -m src.train
Langkah 5: Jalankan Aplikasi Streamlit

---
# Bash
python -m streamlit run app/main.py
Aplikasi web akan otomatis terbuka di browser Anda pada alamat: http://localhost:8501.
