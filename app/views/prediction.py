"""
====================================================================
MODUL VIEW: PREDIKSI RISIKO REAL-TIME (app/views/prediction.py)
====================================================================
Fungsi:
1. Menyediakan form input parametrik pengiriman real-time.
2. Memprediksi probabilitas delay menggunakan model Deep Learning.
3. Mengirimkan catatan transaksi prediksi baru ke st.session_state 
   agar terintegrasi langsung dengan Analytics Dashboard (Real-Time).
====================================================================
"""

import pandas as pd
import streamlit as st


def render_prediction_page(preprocessor, model):
    st.subheader("🔍 Form Input Parametrik Pengiriman Real-Time")
    st.markdown("Isi parameter operasional pengiriman di bawah ini untuk menganalisis potensi risiko keterlambatan:")
    
    # Membagi Form Input menjadi 3 Kolom Visual
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📐 Parametrik Fisik")
        distance_km = st.number_input(
            "Jarak Pengiriman (km)", min_value=1.0, max_value=500.0, value=150.0, step=5.0
        )
        package_weight_kg = st.number_input(
            "Berat Paket (kg)", min_value=0.1, max_value=100.0, value=10.0, step=0.5
        )
        # Estimasi Biaya berbasis korelasi linier jarak
        delivery_cost = distance_km * 5.75
        st.caption(f"💰 **Estimasi Biaya:** Rp {delivery_cost:,.2f}")
        
    with col2:
        st.markdown("#### 🚚 Partner & Lingkungan")
        delivery_partner = st.selectbox(
            "Mitra Ekspedisi", 
            ['delhivery', 'xpressbees', 'shadowfax', 'dhl', 'amazon logistics', 'blue dart', 'fedex', 'ecom express', 'ekart']
        )
        region = st.selectbox(
            "Wilayah Operasional", 
            ['west', 'central', 'east', 'north', 'south']
        )
        weather_condition = st.selectbox(
            "Kondisi Cuaca Rute", 
            ['clear', 'cold', 'rainy', 'foggy', 'hot', 'stormy']
        )
        vehicle_type = st.selectbox(
            "Jenis Kendaraan Pengangkut", 
            ['bike', 'ev van', 'truck', 'van', 'ev bike', 'scooter']
        )
        
    with col3:
        st.markdown("#### 📦 Layanan & Target SLA")
        expected_time_hours = st.number_input(
            "Estimasi Waktu Sampai / SLA (Jam)", min_value=1.0, max_value=72.0, value=8.0, step=1.0
        )
        delivery_mode = st.selectbox(
            "Mode Pengiriman", 
            ['same day', 'express', 'two day', 'standard']
        )
        package_type = st.selectbox(
            "Kategori Barang", 
            ['automobile parts', 'cosmetics', 'groceries', 'electronics', 'clothing', 'documents', 'fragile items', 'pharmacy', 'furniture']
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tombol Eksekusi Prediksi AI
    if st.button("🚀 Eksekusi Analisis Risiko AI", use_container_width=True):
        
        # 1. Susun Data Input Pengguna menjadi Dataframe 1 Baris
        raw_input_df = pd.DataFrame([{
            'delivery_partner': delivery_partner,
            'package_type': package_type,
            'vehicle_type': vehicle_type,
            'delivery_mode': delivery_mode,
            'region': region,
            'weather_condition': weather_condition,
            'distance_km': distance_km,
            'package_weight_kg': package_weight_kg,
            'delivery_time_hours': expected_time_hours,
            'expected_time_hours': expected_time_hours,
            'delivery_rating': 4.0,
            'delivery_cost': delivery_cost,
            'efficiency_ratio': 1.0,
            'tenure': 12
        }])
        
        # 2. Transformasi Data Input via Preprocessor (.pkl)
        transformed_input = preprocessor.transform(raw_input_df)
        
        # 3. Prediksi Probabilitas menggunakan Model Deep Learning (.h5)
        # Array Output: [Prob_Delivered, Prob_Delayed, Prob_Failed]
        predictions = model.predict(transformed_input)[0]
        delay_probability = predictions[1] * 100  # Indeks 1 = Status 'delayed'
        
        # Penentuan Status Hasil Prediksi untuk Dashboard
        predicted_status = 'delayed' if delay_probability > 50 else 'delivered'

        # ------------------------------------------------------------------
        # INTEGRASI SKENARIO B: LOGIKA PENCATATAN TRANSAKSI REAL-TIME
        # ------------------------------------------------------------------
        new_record = {
            'delivery_partner': delivery_partner,
            'package_type': package_type,
            'vehicle_type': vehicle_type,
            'delivery_mode': delivery_mode,
            'region': region,
            'weather_condition': weather_condition,
            'distance_km': distance_km,
            'package_weight_kg': package_weight_kg,
            'delivery_time_hours': expected_time_hours,
            'expected_time_hours': expected_time_hours,
            'delivery_cost': delivery_cost,
            'delivery_status': predicted_status
        }

        # Inisialisasi list pencatatan jika belum ada di memori sesi
        if 'prediction_history' not in st.session_state:
            st.session_state['prediction_history'] = []

        # Tambahkan data baru ke memori sesi Streamlit
        st.session_state['prediction_history'].append(new_record)
        
        # 4. Tampilkan Hasil Prediksi & Visualisasi Probabilitas
        st.markdown("---")
        st.markdown("### 📈 Hasil Inferensi Model AI")
        
        st.write(f"Probabilitas Keterlambatan Pengiriman: **{delay_probability:.1f}%**")
        st.progress(int(delay_probability))
        
        # Notifikasi integrasi Real-Time ke Dashboard
        st.info(
            f"🟢 **Live Feed Updated:** Transaksi prediksi ini otomatis dicatat ke Analytics Dashboard. "
            f"(Total Prediksi Sesi Ini: **{len(st.session_state['prediction_history'])} paket**)"
        )

        # 5. Logika Keputusan Proaktif (Decision Engine & Mitigation Rule)
        if delay_probability > 70:
            st.error(
                f"🔴 **RISIKO TINGGI (Probabilitas Delay: {delay_probability:.1f}%)**\n\n"
                "**Rekomendasi Mitigasi Proaktif:**\n"
                "- Lakukan intervensi vendor atau alihkan mitra logistik ke penyedia berkinerja tinggi (misal: *Delhivery*).\n"
                "- Evaluasi ulang rute untuk menghindari potensi penumpukan pengiriman."
            )
        elif 40 <= delay_probability <= 70:
            st.warning(
                f"🟡 **RISIKO SEDANG (Probabilitas Delay: {delay_probability:.1f}%)**\n\n"
                "**Rekomendasi Mitigasi Proaktif:**\n"
                "- Terapkan *Enhanced Monitoring* pada status perjalanan paket secara berkala.\n"
                "- Berikan notifikasi awal penyesuaian perkiraan SLA kepada pelanggan."
            )
        else:
            st.success(
                f"🟢 **RISIKO RENDAH / OPTIMAL (Probabilitas Delay: {delay_probability:.1f}%)**\n\n"
                "**Status Operational:**\n"
                "- Pengiriman berada dalam kondisi optimal. Lanjutkan pengiriman dengan prosedur standar."
            )