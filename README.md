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



```` ```bash ````

---

### 2. Teks Bersih Siap Salin ke `README.md`

Buka file **`README.md`** di VS Code, hapus teks yang menumpuk, lalu salin (*copy*) dan tempel (*paste*) format yang sudah terpisah sempurna di bawah ini:

```markdown
### 🧠 Arsitektur Machine Learning Pipeline

1. **Feature Engineering & Transformation:**
   - **Numerical Features** (*Robust Scaling / Standardization*): `distance_km`, `package_weight_kg`, `delivery_time_hours`, dll.
   - **Categorical Features** (*One-Hot Encoding*): `delivery_partner`, `region`, `weather_condition`, `vehicle_type`, `package_type`, `delivery_mode`.

2. **Model Architecture:**
   - **Type:** Deep Neural Network (*Multilayer Perceptron / MLP*).
   - **Layers:** Dense Layers with ReLU Activation, Batch Normalization, and Dropout layers (0.2–0.3) for regularization.
   - **Output Layer:** Softmax Multi-class Classifier (`Delivered`, `Delayed`, `Failed`).
   - **Optimization:** Adam Optimizer, Categorical Crossentropy Loss.

---

## ✨ 3. Fitur Utama

- 🔍 **Real-Time Risk Prediction:** Form interaktif untuk memasukkan parameter operasional dan menghitung probabilitas delay dalam hitungan milidetik.
- 🚦 **Proactive Decision Engine:** Aturan mitigasi bisnis dinamis berbasis threshold risiko (Risiko Tinggi >70%, Sedang 40-70%, Optimal <40%).
- 📊 **Dynamic Analytics Dashboard:** Visualisasi distribusi historis dengan kemampuan filter interaktif (Mitra, Wilayah, Cuaca).
- ⚡ **Live-Feed Reactive Data Flow:** Setiap transaksi baru yang dihitung di form prediksi akan otomatis masuk ke memori sesi (`st.session_state`) dan memperbarui KPI serta grafik analitik secara instan.

---

## 🚀 4. Langkah-Langkah Menjalankan Proyek (Step-by-Step)

Ikuti langkah-langkah berikut untuk meng-clone, menyiapkan environment, dan menjalankan aplikasi di komputer lokal Anda:

### Langkah 1: Clone Repositori
Buka terminal Anda dan jalankan perintah berikut:

```bash
git clone https://github.com/shalehh17/Sistem_Pendeteksi_dan_Prediksi_Risiko-.git
cd Sistem_Pendeteksi_dan_Prediksi_Risiko-
```

### Langkah 2: Buat Virtual Environment (Opsional tapi Disarankan)

```bash
# Untuk Windows (Command Prompt / PowerShell)
python -m venv venv
venv\Scripts\activate

# Untuk macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Langkah 3: Install Dependensi

```bash
pip install -r requirements.txt
```

### Langkah 4: Bangun Artefak Model (Jika Belum Tersedia)
Jalankan modul preprocessing data dan training model secara modular:

```bash
# 1. Bangun pipeline preprocessor
python -m src.preprocessing

# 2. Latih model Deep Learning
python -m src.train
```

### Langkah 5: Jalankan Aplikasi Streamlit

```bash
python -m streamlit run app/main.py
```

Aplikasi web akan otomatis terbuka di browser Anda pada alamat: `http://localhost:8501`.

---

## 📈 5. Panduan Penggunaan Aplikasi

1. **Menu: Real-time Prediction**
   - Masukkan informasi rute (*Jarak, Berat Paket, Estimasi Jam*).
   - Tentukan mitra ekspedisi, wilayah operasional, kendaraan, serta kondisi cuaca rute.
   - Klik **"🚀 Eksekusi Analisis Risiko AI"**.
   - Sistem akan menampilkan persentase risiko delay, progress bar, rekomendasi mitigasi bisnis, dan mencatat transaksi ke dasbor analitik secara *live*.

2. **Menu: Analytics Dashboard**
   - Lihat ringkasan KPI (*Total Paket, Tingkat Keterlambatan, Rata-Rata Biaya, dll.*).
   - Gunakan **Panel Filter Data Interaktif** untuk membandingkan performa vendor logistik di berbagai kondisi rute.
   - Pantau *Live Feed Badge* dan lihat transaksi pengujian terbaru pada tabel pratinjau data di baris paling bawah.

---

## 🛠️ 6. Teknologi & Pustaka yang Digunakan

- **Bahasa Pemrograman:** Python 3.9+
- **Antarmuka Web:** Streamlit
- **Machine Learning & Preprocessing:** Scikit-Learn, Joblib
- **Deep Learning Framework:** TensorFlow / Keras
- **Manipulasi & Analisis Data:** Pandas, NumPy
- **Version Control:** Git & GitHub

---

## 👨‍💻 Kontributor

- **Pengembang:** [shalehh17](https://github.com/shalehh17)
- **Kontak / Email:** `shalehuddinzaki84@gmail.com`

-```` ``` ```` 




