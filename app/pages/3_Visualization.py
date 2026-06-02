import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

direktori_halaman = os.path.dirname(os.path.abspath(__file__))
direktori_app_folder = os.path.dirname(direktori_halaman)
direktori_akar_proyek = os.path.dirname(direktori_app_folder)

rute_dataset = os.path.join(direktori_akar_proyek, "dataset", "data_kualitas_udara.csv")
rute_model_km = os.path.join(direktori_akar_proyek, "model", "model_clustering_kmeans.pkl")
rute_scaler_km = os.path.join(direktori_akar_proyek, "model", "scaler_clustering.pkl")
rute_peta = os.path.join(direktori_akar_proyek, "model", "cluster_label_map.pkl")

@st.cache_resource
def muat_aset_biner_visualisasi():
    m_km = joblib.load(rute_model_km)
    s_km = joblib.load(rute_scaler_km)
    p_klas = joblib.load(rute_peta)
    return m_km, s_km, p_klas

rute_css = os.path.join(direktori_akar_proyek, "app", "assets", "style.css")
if os.path.exists(rute_css):
    with open(rute_css, "r") as berkas_css:
        st.markdown(f"<style>{berkas_css.read()}</style>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color: #ffffff; margin-bottom: 10px; font-size: 18px;'>🎯 Navigasi Sistem</h3>", unsafe_allow_html=True)

try:
    model_km, scaler_km, peta_klaster = muat_aset_biner_visualisasi()
except Exception as e:
    st.error(f"Gagal memuat aset biner pkl di halaman visualisasi. Galat: {e}")

st.title("📈 Galeri Visualisasi Hasil Analisis Data Science")
st.write("---")
    
if os.path.exists(rute_dataset):
    bingkai_data = pd.read_csv(rute_dataset)
    
    tab_distribusi, tab_korelasi, tab_kategori, tab_agregat, tab_evaluasi = st.tabs([
        "🔍 Distribusi Polutan", 
        "🔗 Korelasi Zat", 
        "📊 Sebaran Kategori",
        "📉 Tren Agregat Alami",
        "🎯 Validasi Model"
    ])
    
    with tab_distribusi:
        st.subheader("1. Analisis Sebaran Volume Nilai Konsentrasi Parameter Polutan")
        st.write("Silakan klik menu tab di bawah ini untuk melihat kurva kepadatan frekuensi kemunculan polutan secara dinamis.")
        
        t_pm25, t_pm10, t_so2, t_co, t_o3, t_no2 = st.tabs(["PM2.5", "PM10", "SO2", "CO", "O3", "NO2"])
        
        with t_pm25:
            kanvas_pm25, sumbu_pm25 = plt.subplots(figsize=(10, 4.5))
            sns.histplot(bingkai_data['pm25'], kde=True, color='#4b6cb7', ax=sumbu_pm25, edgecolor='black', linewidth=0.5)
            sumbu_pm25.set_title("Grafik Kepadatan Frekuensi Konsentrasi Zat PM2.5", fontsize=11, fontweight='bold')
            sumbu_pm25.set_xlabel("Nilai Kadar PM2.5 (µg/m³)")
            sumbu_pm25.set_ylabel("Intensitas Kemunculan Hari")
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(kanvas_pm25)
            
        with t_pm10:
            kanvas_pm10, sumbu_pm10 = plt.subplots(figsize=(10, 4.5))
            sns.histplot(bingkai_data['pm10'], kde=True, color='#4b6cb7', ax=sumbu_pm10, edgecolor='black', linewidth=0.5)
            sumbu_pm10.set_title("Grafik Kepadatan Frekuensi Konsentrasi Zat PM10", fontsize=11, fontweight='bold')
            sumbu_pm10.set_xlabel("Nilai Kadar PM10 (µg/m³)")
            sumbu_pm10.set_ylabel("Intensitas Kemunculan Hari")
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(kanvas_pm10)
            
        with t_so2:
            kanvas_so2, sumbu_so2 = plt.subplots(figsize=(10, 4.5))
            sns.histplot(bingkai_data['so2'], kde=True, color='#4b6cb7', ax=sumbu_so2, edgecolor='black', linewidth=0.5)
            sumbu_so2.set_title("Grafik Kepadatan Frekuensi Konsentrasi Zat SO2", fontsize=11, fontweight='bold')
            sumbu_so2.set_xlabel("Nilai Kadar SO2 (µg/m³)")
            sumbu_so2.set_ylabel("Intensitas Kemunculan Hari")
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(kanvas_so2)
            
        with t_co:
            kanvas_co, sumbu_co = plt.subplots(figsize=(10, 4.5))
            sns.histplot(bingkai_data['co'], kde=True, color='#4b6cb7', ax=sumbu_co, edgecolor='black', linewidth=0.5)
            sumbu_co.set_title("Grafik Kepadatan Frekuensi Konsentrasi Zat CO", fontsize=11, fontweight='bold')
            sumbu_co.set_xlabel("Nilai Kadar CO (µg/m³)")
            sumbu_co.set_ylabel("Intensitas Kemunculan Hari")
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(kanvas_co)
            
        with t_o3:
            kanvas_o3, sumbu_o3 = plt.subplots(figsize=(10, 4.5))
            sns.histplot(bingkai_data['o3'], kde=True, color='#4b6cb7', ax=sumbu_o3, edgecolor='black', linewidth=0.5)
            sumbu_o3.set_title("Grafik Kepadatan Frekuensi Konsentrasi Zat O3", fontsize=11, fontweight='bold')
            sumbu_o3.set_xlabel("Nilai Kadar O3 (µg/m³)")
            sumbu_o3.set_ylabel("Intensitas Kemunculan Hari")
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(kanvas_o3)
            
        with t_no2:
            kanvas_no2, sumbu_no2 = plt.subplots(figsize=(10, 4.5))
            sns.histplot(bingkai_data['no2'], kde=True, color='#4b6cb7', ax=sumbu_no2, edgecolor='black', linewidth=0.5)
            sumbu_no2.set_title("Grafik Kepadatan Frekuensi Konsentrasi Zat NO2", fontsize=11, fontweight='bold')
            sumbu_no2.set_xlabel("Nilai Kadar NO2 (µg/m³)")
            sumbu_no2.set_ylabel("Intensitas Kemunculan Hari")
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(kanvas_no2)
        
    with tab_korelasi:
        st.subheader("2. Matriks Korelasi Linear Pearson Antar Parameter Polutan")
        st.write("Grafik Peta Panas dengan topeng segitiga atas ini membuktikan kekuatan hubungan linear antar-zat polutan secara bersih tanpa redundansi data.")
        
        kanvas_corr, sumbu_corr = plt.subplots(figsize=(11, 7))
        matriks_korelasi = bingkai_data[['pm25', 'pm10', 'so2', 'co', 'o3', 'no2']].corr()
        topeng_segitiga = np.triu(np.ones_like(matriks_korelasi, dtype=bool))
        
        sns.heatmap(
            data=matriks_korelasi,
            mask=topeng_segitiga,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=1.5,
            linecolor="white",
            ax=sumbu_corr,
            annot_kws={
                "size": 11,
                "weight": "bold"
            },
            cbar_kws={
                "shrink": 0.8,
                "label": "Kekuatan Korelasi"
            }
        )
        sumbu_corr.set_title("Matriks Korelasi Antar Parameter Polutan", fontsize=15, fontweight="bold", pad=18)
        plt.xticks(rotation=25, ha="right", fontsize=11)
        plt.yticks(rotation=0, fontsize=11)
        plt.tight_layout()
        st.pyplot(kanvas_corr)
        
    with tab_kategori:
        st.subheader("3. Sebaran Parameter Polutan per Kategori Kualitas Udara")
        st.write("Analisis diagram kotak untuk melihat rentang nilai distribusi polutan berdasarkan status formal hukum lingkungan pemerintah.")
        
        kandidat_kat = ['category', 'categori', 'kategori', 'Kategori', 'CATEGORY']
        kolom_kat = None
        for k in kandidat_kat:
            if k in bingkai_data.columns:
                kolom_kat = k
                break
        
        if kolom_kat is not None:
            t_box_pm25, t_box_pm10, t_box_so2, t_box_co, t_box_o3, t_box_no2 = st.tabs(["PM2.5", "PM10", "SO2", "CO", "O3", "NO2"])
            
            with t_box_pm25:
                kanvas_box1, sumbu_box1 = plt.subplots(figsize=(10, 5))
                sns.boxplot(x=kolom_kat, y='pm25', data=bingkai_data, palette="Set2", ax=sumbu_box1)
                sumbu_box1.set_title("Rentang Nilai Konsentrasi PM2.5 Berdasarkan Kategori Kualitas Udara", fontsize=11, fontweight='bold')
                sumbu_box1.set_xlabel("Kategori Kualitas Udara Resmi")
                sumbu_box1.set_ylabel("Kadar PM2.5 (µg/m³)")
                st.pyplot(kanvas_box1)
                
            with t_box_pm10:
                kanvas_box2, sumbu_box2 = plt.subplots(figsize=(10, 5))
                sns.boxplot(x=kolom_kat, y='pm10', data=bingkai_data, palette="Set2", ax=sumbu_box2)
                sumbu_box2.set_title("Rentang Nilai Konsentrasi PM10 Berdasarkan Kategori Kualitas Udara", fontsize=11, fontweight='bold')
                sumbu_box2.set_xlabel("Kategori Kualitas Udara Resmi")
                sumbu_box2.set_ylabel("Kadar PM10 (µg/m³)")
                st.pyplot(kanvas_box2)
                
            with t_box_so2:
                kanvas_box3, sumbu_box3 = plt.subplots(figsize=(10, 5))
                sns.boxplot(x=kolom_kat, y='so2', data=bingkai_data, palette="Set2", ax=sumbu_box3)
                sumbu_box3.set_title("Rentang Nilai Konsentrasi SO2 Berdasarkan Kategori Kualitas Udara", fontsize=11, fontweight='bold')
                sumbu_box3.set_xlabel("Kategori Kualitas Udara Resmi")
                sumbu_box3.set_ylabel("Kadar SO2 (µg/m³)")
                st.pyplot(kanvas_box3)
                
            with t_box_co:
                kanvas_box4, sumbu_box4 = plt.subplots(figsize=(10, 5))
                sns.boxplot(x=kolom_kat, y='co', data=bingkai_data, palette="Set2", ax=sumbu_box4)
                sumbu_box4.set_title("Rentang Nilai Konsentrasi CO Berdasarkan Kategori Kualitas Udara", fontsize=11, fontweight='bold')
                sumbu_box4.set_xlabel("Kategori Kualitas Udara Resmi")
                sumbu_box4.set_ylabel("Kadar CO (µg/m³)")
                st.pyplot(kanvas_box4)
                
            with t_box_o3:
                kanvas_box5, sumbu_box5 = plt.subplots(figsize=(10, 5))
                sns.boxplot(x=kolom_kat, y='o3', data=bingkai_data, palette="Set2", ax=sumbu_box5)
                sumbu_box5.set_title("Rentang Nilai Konsentrasi O3 Berdasarkan Kategori Kualitas Udara", fontsize=11, fontweight='bold')
                sumbu_box5.set_xlabel("Kategori Kualitas Udara Resmi")
                sumbu_box5.set_ylabel("Kadar O3 (µg/m³)")
                st.pyplot(kanvas_box5)
                
            with t_box_no2:
                kanvas_box6, sumbu_box6 = plt.subplots(figsize=(10, 5))
                sns.boxplot(x=kolom_kat, y='no2', data=bingkai_data, palette="Set2", ax=sumbu_box6)
                sumbu_box6.set_title("Rentang Nilai Konsentrasi NO2 Berdasarkan Kategori Kualitas Udara", fontsize=11, fontweight='bold')
                sumbu_box6.set_xlabel("Kategori Kualitas Udara Resmi")
                sumbu_box6.set_ylabel("Kadar NO2 (µg/m³)")
                st.pyplot(kanvas_box6)
        else:
            st.warning("Kolom kategori tidak ditemukan di dalam dataset untuk membuat boxplot.")
            
    with tab_agregat:
        st.subheader("4. Analisis Nilai Maksimum Parameter Polutan Global")
        st.write("Melihat rekam jejak angka tertinggi yang pernah dicapai oleh masing-masing zat polutan di atmosfer.")
        
        kanvas_bar, sumbu_bar = plt.subplots(figsize=(10, 5.5))
        data_fitur_max = bingkai_data[['pm25', 'pm10', 'so2', 'co', 'o3', 'no2']].max().reset_index()
        data_fitur_max.columns = ['Parameter Polutan', 'Nilai Maksimum']
        data_fitur_max = data_fitur_max.sort_values(by='Nilai Maksimum', ascending=False)
        
        grafik_batang = sns.barplot(
            x='Parameter Polutan', 
            y='Nilai Maksimum', 
            data=data_fitur_max, 
            palette='flare', 
            ax=sumbu_bar,
            edgecolor='black',
            linewidth=0.5
        )
        
        for kontainer in grafik_batang.containers:
            sumbu_bar.bar_label(kontainer, fmt='%.1f', padding=5, fontweight='bold', fontsize=10)
            
        sumbu_bar.set_title("Manifes Nilai Konsentrasi Maksimum Seluruh Parameter Polutan", fontsize=12, fontweight='bold', pad=12)
        sumbu_bar.set_xlabel("Jenis Zat Parameter Polutan")
        sumbu_bar.set_ylabel("Nilai Kadar Tertinggi (µg/m³)")
        sumbu_bar.set_ylim(0, data_fitur_max['Nilai Maksimum'].max() * 1.15)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(kanvas_bar)
        
    with tab_evaluasi:
        st.subheader("5. Validasi Matriks Silang Model K-Means Clustering")
        st.write("Grafik pembuktian seberapa selaras batasan klaster alami bentukan model K-Means dengan kategori asli milik pemerintah.")
        
        urutan_baris = ['BAIK', 'SEDANG', 'TIDAK SEHAT', 'SANGAT TIDAK SEHAT']
        urutan_kolom = [
            "Cluster Baik (Polusi Rendah)",
            "Cluster Sedang (Polusi Moderat)",
            "Cluster Tidak Sehat (Polusi Tinggi)",
            "Cluster Sangat Tidak Sehat (Polusi Ekstrem)"
        ]
        
        if kolom_kat is not None:
            df_visual = bingkai_data.copy()
            
            if 'label_cluster' not in df_visual.columns:
                data_fitur = df_visual[['pm25', 'pm10', 'so2', 'co', 'o3', 'no2']]
                data_terpangkas = scaler_km.transform(data_fitur)
                prediksi_cluster_array = model_km.predict(data_terpangkas)
                df_visual['label_cluster'] = [peta_klaster[int(c)] for c in prediksi_cluster_array]
            
            baris_aktif = [b for b in urutan_baris if b in df_visual[kolom_kat].unique()]
            kolom_aktif = [k for k in urutan_kolom if k in df_visual['label_cluster'].unique()]
            
            tabulasi_heatmap = pd.crosstab(df_visual[kolom_kat], df_visual['label_cluster']).reindex(index=baris_aktif, columns=kolom_aktif).fillna(0).astype(int)
            
            kanvas_heat, sumbu_heat = plt.subplots(figsize=(9, 5.5))
            sns.heatmap(
                data=tabulasi_heatmap, 
                annot=True, 
                fmt='d', 
                cmap='YlGnBu', 
                ax=sumbu_heat,
                linewidths=1.5,
                linecolor='white',
                annot_kws={"size": 11, "weight": "bold"}
            )
            sumbu_heat.set_title("Matriks Crosstab Validasi Kategori Pemerintah vs Klaster K-Means", fontsize=11, fontweight='bold', pad=12)
            sumbu_heat.set_xlabel("Kelompok Label Cluster Alami K-Means")
            sumbu_heat.set_ylabel("Kategori Kualitas Udara Aktual Pemerintah")
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            st.pyplot(kanvas_heat)
        else:
            st.warning("Sistem检测到列'category' aktual tidak tersedia di dalam file CSV Anda untuk menggambar Crosstab Heatmap.")
else:
    st.error("Gagal menggambar galeri visualisasi karena berkas data utama csv tidak ditemukan.")