"""
====================================================================
PIPELINE PEMPROSESAN DATA LOGISTIK (DATA PREPROCESSING & PIPELINE)
====================================================================
Tujuan:
1. Membaca & membersihkan data mentah pengiriman logistik.
2. Membuat fitur baru (Feature Engineering) untuk meningkatkan akurasi AI.
3. Mengubah teks kategorikal menjadi angka (One-Hot Encoding) & menyamakan skala angka (Standard Scaling).
4. Membagi data menjadi data latih (Train) dan data uji (Test).
5. Menyimpan alat pemroses (Preprocessor) untuk digunakan saat prediksi di Streamlit.
====================================================================
"""

import os
import re
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Impor lokasi folder dari modul utils
from src.utils import DATA_PATH, PREPROCESSOR_PATH, MODEL_DIR


# ------------------------------------------------------------------
# TAHAP 1: HELPER FUNCTION (PEMBERSIH KUSTOM WAKTU)
# ------------------------------------------------------------------
def parse_hours(val):
    """
    Fungsi Pembantu: Mengonversi data waktu bertipe timestamp/string 
    menjadi nilai numerik jam yang bersih.
    Contoh: '1970-01-01 00:00:00.000000008' -> 8.0 jam
    """
    if pd.isna(val):
        return 0.0
    
    s = str(val)
    try:
        # Jika nilai sudah berupa angka biasa
        return float(s)
    except ValueError:
        pass
    
    # Mengambil digit angka paling belakang menggunakan Regular Expression (Regex)
    match = re.search(r'(\d+)$', s)
    if match:
        return float(match.group(1))
    
    return 0.0


# ------------------------------------------------------------------
# TAHAP 2: PIPELINE UTAMA PREPROCESSING DATA
# ------------------------------------------------------------------
def load_and_preprocess_data():
    print("📥 [1/6] Memuat dataset mentah dari disk...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"❌ Dataset tidak ditemukan di: {DATA_PATH}. "
            "Pastikan file 'Delivery_Logistics.csv' diletakkan di dalam folder 'data/'."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"   -> Berhasil memuat {len(df):,} baris data dan {len(df.columns)} kolom.")

    # --------------------------------------------------------------
    # TAHAP 3: DATA CLEANING (PEMBERSIHAN DATA WAKTU)
    # --------------------------------------------------------------
    print("🧹 [2/6] Membersihkan format kolom waktu (delivery & expected hours)...")
    df['delivery_time_hours'] = df['delivery_time_hours'].apply(parse_hours)
    df['expected_time_hours'] = df['expected_time_hours'].apply(parse_hours)

    # --------------------------------------------------------------
    # TAHAP 4: FEATURE ENGINEERING (REKAYASA FITUR BARU)
    # --------------------------------------------------------------
    print("💡 [3/6] Membuat fitur baru (Feature Engineering)...")
    
    # Rasio Efisiensi = Waktu Aktual / Waktu Estimasi
    if 'efficiency_ratio' not in df.columns:
        df['efficiency_ratio'] = df['delivery_time_hours'] / (df['expected_time_hours'] + 1e-5)
    
    if 'tenure' not in df.columns:
        df['tenure'] = 12

    # --------------------------------------------------------------
    # TAHAP 5: PEMISAHAN FITUR INPUT (X) DAN TARGET (y)
    # --------------------------------------------------------------
    print("✂️ [4/6] Memisahkan Fitur Input (X) dan Label Target (y)...")
    
    # Hapus kolom identitas & kolom redundan pembawa kebocoran data (Data Leakage)
    X = df.drop(columns=['delivery_status', 'delivery_id', 'is_delayed', 'delayed'], errors='ignore')
    
    # Target Encoding: Mengubah label status teks menjadi angka untuk model AI
    # 0 = delivered (berhasil), 1 = delayed (terlambat), 2 = failed (gagal)
    target_map = {'delivered': 0, 'delayed': 1, 'failed': 2}
    y = df['delivery_status'].map(target_map)

    # Memisahkan nama kolom numerik dan kategorikal
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"   -> Jumlah Fitur Numerik: {len(num_features)} kolom")
    print(f"   -> Jumlah Fitur Kategorikal (Teks): {len(cat_features)} kolom")

    # --------------------------------------------------------------
    # TAHAP 6: MEMBANGUN SKLEARN PREPROCESSOR PIPELINE
    # --------------------------------------------------------------
    print("⚙️ [5/6] Membangun Pipeline Transformasi (Scaling & One-Hot Encoding)...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )

    # --------------------------------------------------------------
    # TAHAP 7: TRAIN-TEST SPLIT & TRANSFORMASI DATA
    # --------------------------------------------------------------
    print("📊 [6/6] Membagi data (80% Latih, 20% Uji) & Menjalankan Transformasi...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit dan Transformasi pada Data Latih, Transformasi pada Data Uji
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # --------------------------------------------------------------
    # TAHAP 8: MENYIMPAN ARTEFAK PREPROCESSOR (.PKL)
    # --------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    
    print("-" * 65)
    print("🎉 PEMPROSESAN DATA SELESAI!")
    print(f"   -> Preprocessor berhasil disimpan di: {PREPROCESSOR_PATH}")
    print(f"   -> Ukuran Data Latih (X_train): {X_train_transformed.shape}")
    print(f"   -> Ukuran Data Uji (X_test): {X_test_transformed.shape}")
    print("-" * 65)

    return X_train_transformed, X_test_transformed, y_train, y_test


if __name__ == "__main__":
    load_and_preprocess_data()