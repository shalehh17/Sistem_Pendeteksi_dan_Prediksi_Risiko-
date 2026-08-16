"""
====================================================================
MODUL UTAMA ENTRY POINT (app/main.py)
====================================================================
Tujuan:
1. Memuat artefak AI (Preprocessor, Model, & Dataset) ke memori.
2. Mengatur konfigurasi halaman & navigasi sidebar Streamlit.
3. Menggabungkan data historis CSV dengan data prediksi baru (Live Feed).
4. Memanggil modul tampilan (views) sesuai halaman yang dipilih pengguna.
====================================================================
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf

# Tambahkan direktori root proyek ke sistem path Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import lokasi file dari modul utilitas internal
from src.utils import PREPROCESSOR_PATH, DL_MODEL_PATH, DATA_PATH

# Import modul tampilan (views) dari folder app/views/
from app.views.prediction import render_prediction_page
from app.views.dashboard import render_dashboard_page


# ------------------------------------------------------------------
# TAHAP 1: KONFIGURASI TAMPILAN HALAMAN STREAMLIT
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Logistics AI Concierge",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------------
# TAHAP 2: MEMUAT ARTEFAK AI & DATASET (RESOURCE CACHING)
# ------------------------------------------------------------------
@st.cache_resource
def load_machine_learning_assets():
    """
    Fungsi Cache: Memuat model dan dataset hanya satu kali ke dalam memori 
    agar aplikasi web berjalan sangat cepat dan tidak lambat saat diklik.
    """
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = tf.keras.models.load_model(DL_MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    return preprocessor, model, df


# Penanganan Proteksi Error jika Artefak Model Belum Dibuat
try:
    preprocessor, model, df = load_machine_learning_assets()
except Exception as e:
    st.error("❌ **Gagal Memuat Artefak AI!**")
    st.warning(
        "Sistem tidak menemukan file model terlatih (`.pkl` / `.h5`).\n\n"
        "**Solusi:** Silakan jalankan perintah berikut di terminal Anda terlebih dahulu:\n"
        "1. `python -m src.preprocessing` (untuk membuat preprocessor)\n"
        "2. `python -m src.train` (untuk melatih model AI)"
    )
    st.info(f"Detail Pesan Error: {e}")
    st.stop()


# ------------------------------------------------------------------
# TAHAP 3: HEADER APLIKASI WEB & NAVIGASI SIDEBAR
# ------------------------------------------------------------------
st.title("🚚 Logistics AI Concierge")
st.caption("The Intelligent Route — Risk Prediction & Mitigation System (Model Accuracy > 95%)")
st.markdown("---")

st.sidebar.header("🕹️ Navigasi Sistem")
menu = st.sidebar.radio(
    "Pilih Fitur Aplikasi:",
    ["Real-time Prediction", "Analytics Dashboard"]
)

# Hitung jumlah transaksi prediksi baru dari memori sesi (session_state)
new_predictions_count = len(st.session_state.get('prediction_history', []))

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Informasi Model & Live Feed:**\n"
    "- **Arsitektur:** Deep Neural Network (MLP)\n"
    "- **Target Evaluasi:** Delivered, Delayed, Failed\n"
    "- **Status Model:** Teroptimasi (~97.7% Accuracy)\n"
    f"- **Prediksi Live Sesi Ini:** **{new_predictions_count} paket**"
)


# ------------------------------------------------------------------
# TAHAP 4: ROUTING & PEMANGGILAN VIEW (SKENARIO B REAL-TIME MERGE)
# ------------------------------------------------------------------
if menu == "Real-time Prediction":
    # Memanggil tampilan dari app/views/prediction.py
    render_prediction_page(preprocessor, model)

elif menu == "Analytics Dashboard":
    # 1. Buat salinan dataset historis dari CSV
    combined_df = df.copy()

    # 2. Jika ada transaksi prediksi baru dari customer, gabungkan ke dataset historis
    if 'prediction_history' in st.session_state and len(st.session_state['prediction_history']) > 0:
        new_data_df = pd.DataFrame(st.session_state['prediction_history'])
        combined_df = pd.concat([combined_df, new_data_df], ignore_index=True)

    # 3. Kirim dataset ter-update (historis + live feed) ke Analytics Dashboard
    render_dashboard_page(combined_df)