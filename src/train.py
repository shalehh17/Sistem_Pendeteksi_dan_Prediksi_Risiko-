"""
====================================================================
MODUL PELATIHAN MODEL DEEP LEARNING (MODEL TRAINING PIPELINE)
====================================================================
Tujuan:
1. Memuat data pengiriman yang sudah diproses (Train & Test Sets).
2. Membangun arsitektur Neural Network bertingkat (Sequential Model).
3. Mengonfigurasi strategi pengoptimalan (Optimizer & Callbacks).
4. Melatih model secara bertahap (Training Iteration/Epochs).
5. Evaluasi performa akhir dan menyimpan bobot model (.h5) ke disk.
====================================================================
"""

import os
import sys

# ------------------------------------------------------------------
# SETUP JALUR SYSTEM 
# ------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import fungsi pengolah data & path dari modul internal
from src.preprocessing import load_and_preprocess_data
from src.utils import DL_MODEL_PATH


# ------------------------------------------------------------------
# TAHAP 1: KONSTRUKSI ARSITEKTUR NEURAL NETWORK
# ------------------------------------------------------------------
def build_neural_network(input_dimension: int) -> Sequential:
    """
    Membuat arsitektur Deep Learning Multi-Layer Perceptron (MLP)
    dengan nama layer unik untuk mencegah ValueError duplicate layer names.
    
    Penjelasan Layer :
    - Dense: Layer neuron terhubung penuh (Fully Connected).
    - BatchNormalization: Memproses nilai masukan antar-layer agar stabil & cepat belajar.
    - Dropout: Mematikan sebagian neuron secara acak untuk mencegah 'Overfitting'.
    - Softmax: Layer output yang mengubah skor menjadi nilai probabilitas (0% - 100%).
    """
    model = Sequential([
        # Hidden Layer 1: Menerima fitur numerik + kategorikal One-Hot
        Dense(128, activation='relu', input_shape=(input_dimension,), name="dense_input"),
        BatchNormalization(name="bn_1"),
        Dropout(0.3, name="dropout_1"),  # Mematikan 30% neuron secara acak
        
        # Hidden Layer 2: Penyerapan pola lebih dalam
        Dense(64, activation='relu', name="dense_hidden_1"),
        BatchNormalization(name="bn_2"),
        Dropout(0.2, name="dropout_2"),  # Mematikan 20% neuron
        
        # Hidden Layer 3: Pengikatan pola sebelum keputusan
        Dense(32, activation='relu', name="dense_hidden_2"),
        
        # Output Layer: 3 Kelas Target -> Delivered (0), Delayed (1), Failed (2)
        Dense(3, activation='softmax', name="dense_output")
    ])
    
    return model


# ------------------------------------------------------------------
# TAHAP 2: PIPELINE PELATIHAN MODEL (TRAINING PIPELINE)
# ------------------------------------------------------------------
def train_model():
    print("\n" + "=" * 65)
    print("🧠 TAHAP 1: MEMUAT DATASET TERTRANSFORMASI")
    print("=" * 65)
    
    # 1. Load Data dari Preprocessing Pipeline
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    input_dim = X_train.shape[1]
    print(f"\n💡 Total Dimensi Fitur Masukan (Input Features): {input_dim} kolom")

    print("\n" + "=" * 65)
    print("🏗️ TAHAP 2: MEMBANGUN & MENGOMPILASI ARSITEKTUR AI")
    print("=" * 65)
    
    # Reset memori Keras agar tidak ada sisa nama layer dari sesi sebelumnya
    tf.keras.backend.clear_session()
    
    # 2. Inisialisasi Arsitektur
    model = build_neural_network(input_dimension=input_dim)
    
    # 3. Kompilasi Model (Menentukan Pengoptimasi & Rumus Kerugian)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Tampilkan Ringkasan Arsitektur Model
    model.summary()

    # 4. Pengaturan Safeguard (Callbacks)
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', 
            patience=10, 
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=4, 
            verbose=1
        )
    ]

    print("\n" + "=" * 65)
    print("🚀 TAHAP 3: MEMULAI PROSES PELATIHAN (TRAINING EPOCHS)")
    print("=" * 65)
    
    # 5. Fit / Latih Model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "=" * 65)
    print("📊 TAHAP 4: EVALUASI PERFORMA MODEL PADA DATA UJI")
    print("=" * 65)
    
    # 6. Evaluasi Akurasi Akhir
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   -> Testing Loss     : {loss:.4f}")
    print(f"   -> Testing Accuracy : {accuracy * 100:.2f}% (Target: > 95%)")

    # 7. Simpan Artefak Model Kecerdasan Buatan (.h5)
    os.makedirs(os.path.dirname(DL_MODEL_PATH), exist_ok=True)
    model.save(DL_MODEL_PATH)
    
    print("-" * 65)
    print("🎉 PELATIHAN MODEL SELESAI!")
    print(f"   -> Otak AI berhasil disimpan di: {DL_MODEL_PATH}")
    print("-" * 65 + "\n")


if __name__ == "__main__":
    train_model()