"""
====================================================================
MODUL VIEW: INTERACTIVE ANALYTICS DASHBOARD (app/views/dashboard.py)
====================================================================
Fungsi:
1. Menerima data gabungan (Historis CSV + Live Feed Prediksi Baru).
2. Menyajikan badge indikator Live Feed Data secara real-time.
3. Memudahkan filter interaktif berdasarkan pilihan customer.
4. Menghitung kartu metrik KPI dan visualisasi grafik secara dinamis.
====================================================================
"""

import pandas as pd
import streamlit as st


def render_dashboard_page(df):
    st.subheader("📊 Analytics & Strategic Insights Dashboard")

    # ------------------------------------------------------------------
    # TAHAP 1: INDIKATOR BADGE LIVE FEED DATA
    # ------------------------------------------------------------------
    new_data_count = len(st.session_state.get('prediction_history', []))
    if new_data_count > 0:
        st.success(
            f"🟢 **Live Feed Aktif:** Dasbor menggabungkan data historis + **{new_data_count} data prediksi baru** "
            "dari transaksi sesi ini."
        )

    st.markdown("Gunakan panel filter di bawah ini untuk menganalisis performa operasional logistik secara interaktif:")

    # ------------------------------------------------------------------
    # TAHAP 2: PANEL FILTER INTERAKTIF (PILIKAN CUSTOMER)
    # ------------------------------------------------------------------
    with st.expander("🎛️ **Panel Filter Data Interaktif**", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)

        # Filter 1: Mitra Ekspedisi
        all_partners = sorted(df['delivery_partner'].dropna().unique().tolist()) if 'delivery_partner' in df.columns else []
        selected_partners = col_f1.multiselect(
            "Pilih Mitra Ekspedisi:",
            options=all_partners,
            default=all_partners,
            help="Pilih satu atau beberapa mitra ekspedisi."
        )

        # Filter 2: Wilayah Operasional
        all_regions = sorted(df['region'].dropna().unique().tolist()) if 'region' in df.columns else []
        selected_regions = col_f2.multiselect(
            "Pilih Wilayah Operasional:",
            options=all_regions,
            default=all_regions,
            help="Pilih wilayah operasional pengiriman."
        )

        # Filter 3: Kondisi Cuaca Rute
        all_weather = sorted(df['weather_condition'].dropna().unique().tolist()) if 'weather_condition' in df.columns else []
        selected_weather = col_f3.multiselect(
            "Pilih Kondisi Cuaca Rute:",
            options=all_weather,
            default=all_weather,
            help="Filter berdasarkan kondisi cuaca rute."
        )

    # ------------------------------------------------------------------
    # TAHAP 3: PROSES FILTER DATA DINAMIS
    # ------------------------------------------------------------------
    filtered_df = df.copy()

    if selected_partners and 'delivery_partner' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['delivery_partner'].isin(selected_partners)]

    if selected_regions and 'region' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]

    if selected_weather and 'weather_condition' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['weather_condition'].isin(selected_weather)]

    # Penanganan Proteksi jika Hasil Filter Kosong
    if filtered_df.empty:
        st.warning("⚠️ Tidak ditemukan data pengiriman yang sesuai dengan kriteria filter Anda. Silakan sesuaikan kembali pilihan filter.")
        return

    st.markdown("---")

    # ------------------------------------------------------------------
    # TAHAP 4: KALKULASI METRIK & KPI DINAMIS
    # ------------------------------------------------------------------
    total_paket = len(filtered_df)

    # Hitung persentase delay secara real-time
    if 'delivery_status' in filtered_df.columns:
        delayed_count = len(filtered_df[filtered_df['delivery_status'] == 'delayed'])
        delay_rate = (delayed_count / total_paket * 100) if total_paket > 0 else 0.0
    else:
        delay_rate = 0.0

    avg_distance = filtered_df['distance_km'].mean() if 'distance_km' in filtered_df.columns else 0.0
    avg_cost = filtered_df['delivery_cost'].mean() if 'delivery_cost' in filtered_df.columns else 0.0

    # Menampilkan 4 Kartu Metrik Ringkasan Dinamis
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Paket Terfilter", f"{total_paket:,} Paket")
    col_m2.metric("Tingkat Keterlambatan (Delay)", f"{delay_rate:.1f}%")
    col_m3.metric("Rata-Rata Jarak Rute", f"{avg_distance:.1f} km")
    col_m4.metric("Rata-Rata Biaya", f"Rp {avg_cost:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # TAHAP 5: VISUALISASI DINAMIS (BAR CHARTS)
    # ------------------------------------------------------------------
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Proporsi Status Pengiriman (Historis + Live Feed)")
        if 'delivery_status' in filtered_df.columns:
            status_counts = filtered_df['delivery_status'].value_counts()
            st.bar_chart(status_counts)
            st.caption("Grafik otomatis memperbarui distribusi status mencakup data prediksi baru.")

    with col_b:
        st.markdown("#### Distribusi Keterlambatan per Mitra Ekspedisi")
        if 'delivery_status' in filtered_df.columns and 'delivery_partner' in filtered_df.columns:
            delayed_df = filtered_df[filtered_df['delivery_status'] == 'delayed']
            partner_delay_counts = delayed_df['delivery_partner'].value_counts()

            if not partner_delay_counts.empty:
                st.bar_chart(partner_delay_counts)
                st.caption("Menampilkan frekuensi paket delayed per mitra berdasarkan filter pilihan.")
            else:
                st.success("🟢 Tidak ada paket yang mengalami delay pada kombinasi filter ini.")

    # ------------------------------------------------------------------
    # TAHAP 6: STRATEGIC INSIGHTS & PRATINJAU DATA GABUNGAN
    # ------------------------------------------------------------------
    st.markdown("---")
    col_i1, col_i2 = st.columns([1, 1])

    with col_i1:
        st.markdown("#### 💡 Strategic Vendor Insights")
        st.info(
            "🏆 **Mitra Performa Terbaik:**\n"
            "- **Delhivery** mencatatkan tingkat ketepatan waktu pengiriman tertinggi (~75.2%)."
        )
        st.warning(
            "⚠️ **Mitra Berisiko Keterlambatan Tinggi:**\n"
            "- **Shadowfax & Xpressbees** membutuhkan evaluasi operasional karena memiliki proporsi delay historis >22%."
        )

    with col_i2:
        st.markdown("#### 📋 Pratinjau Dataset Terfilter (Data Terakhir)")
        st.dataframe(filtered_df.tail(10), use_container_width=True)
        st.caption(f"Menampilkan 10 baris terakhir dari total {len(filtered_df):,} baris data terfilter (transaksi live terbaru berada di posisi paling bawah).")