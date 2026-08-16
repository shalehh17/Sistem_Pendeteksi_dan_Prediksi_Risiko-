"""
====================================================================
MODULE: SETUP NAVIGASI ALAMAT & PATH PROYEK (UTILS)
====================================================================
Fungsi:
Modul ini bertindak sebagai "Peta Navigasi Otomatis". 
Modul ini menentukan lokasi folder utama proyek secara dinamis agar 
kode dapat berjalan di komputer mana pun tanpa masalah "File Not Found".
====================================================================
"""

from pathlib import Path

# ------------------------------------------------------------------
# 1. MENENTUKAN ALAMAT UTAMA PROYEK (ROOT DIRECTORY)
# ------------------------------------------------------------------
# Path(__file__) adalah lokasi file utils.py saat ini.
# .resolve() mengubahnya menjadi jalur absolut.
# .parents[1] naik 2 tingkat folder ke atas untuk menemukan direktori utama proyek.
ROOT_DIR = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------
# 2. NAVIGASI DATASET (DATA INGESTION PATH)
# ------------------------------------------------------------------
# Lokasi file dataset mentah yang digunakan untuk latihan model
DATA_DIR = ROOT_DIR / "data"
DATA_PATH = DATA_DIR / "Delivery_Logistics.csv"

# ------------------------------------------------------------------
# 3. NAVIGASI ARTEFAK MODEL (MODEL ARTIFACTS PATH)
# ------------------------------------------------------------------
# Tempat menyimpan hasil pembelajaran mesin (model & pemroses data)
MODEL_DIR = ROOT_DIR / "models"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"  # Pipet transformasi data
DL_MODEL_PATH = MODEL_DIR / "deep_learning_model.h5"  # Otak kecerdasan AI (Keras)

# ------------------------------------------------------------------
# 4. PEMERIKSAAN OTOMATIS (HELPER FUNCTION UNTUK PEMULA)
# ------------------------------------------------------------------
def init_project_structure():
    """Memastikan folder penting sudah dibuat secara otomatis di komputer."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Jalankan pemeriksaan folder saat file utils dipanggil
init_project_structure()