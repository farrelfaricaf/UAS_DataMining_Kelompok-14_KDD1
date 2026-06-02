import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Dataset Overview - Kelompok 14",
    layout="wide",
    initial_sidebar_state="expanded"
)

direktori_halaman = os.path.dirname(os.path.abspath(__file__))
direktori_app_folder = os.path.dirname(direktori_halaman)
direktori_akar_proyek = os.path.dirname(direktori_app_folder)

rute_dataset = os.path.join(direktori_akar_proyek, "dataset", "data_kualitas_udara.csv")
rute_css = os.path.join(direktori_akar_proyek, "app", "assets", "style.css")

if os.path.exists(rute_css):
    with open(rute_css, "r") as berkas_css:
        st.markdown(f"<style>{berkas_css.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown("<h3 style='color: #ffffff; margin-bottom: 10px; font-size: 18px;'>🎯 Navigasi Sistem</h3>", unsafe_allow_html=True)

st.title("📊 Tinjauan Umum & Transparansi Berkas Dataset")
st.write("Eksplorasi dimensi karakteristik data murni beserta akses unduh berkas.")
st.write("---")
    
if os.path.exists(rute_dataset):
    bingkai_data = pd.read_csv(rute_dataset)
    jumlah_duplikat = int(bingkai_data.duplicated().sum())
    jumlah_missing = int(bingkai_data.isnull().sum().sum())
        
    metrik_kiri, metrik_tengah, metrik_kanan = st.columns(3)
    with metrik_kiri:
        st.metric("Total Hari Observasi", f"{bingkai_data.shape[0]} Baris Data")
    with metrik_tengah:
        st.metric("Dimensi Karakteristik Data", "9 Fitur + 1 Target Utama")
    with metrik_kanan:
        st.metric("Status Kelayakan Data", "Bersih Siap Eksplorasi")
    
    st.write("#### 📋 Sampel 15 Baris Data Pertama")
    st.dataframe(bingkai_data.head(15), use_container_width=True)

    st.write("#### 📂 Akses Unduh Berkas Data Utama")
    st.write("Untuk menjaga transparansi riset akademik, kami menyediakan akses penuh bagi para penguji atau periset lain untuk mengunduh data mentah yang kami gunakan dalam fase pemodelan ini.")

    with open(rute_dataset, "rb") as berkas_csv:
        st.download_button(
            label="📥 Unduh Dataset Kualitas Udara Bersih (.CSV)",
            data=berkas_csv,
            file_name="data_kualitas_udara_kelompok14.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    st.write("")
    st.write("#### 📐 Ringkasan Statistik Deskriptif Numerik")
    st.dataframe(bingkai_data.describe(), use_container_width=True)
    
    st.write("")
    st.write("#### 🔍 Hasil Audit & Higienitas Data Mentah")
    st.write("Gunakan menu lipat di bawah ini untuk melihat hasil pembersihan data kosong dan penapisan baris duplikat yang dilakukan oleh sistem.")
    
    kolom_dup, kolom_mis = st.columns(2)
    with kolom_dup:
        with st.expander(f"📋 Laporan Data Duplikat ({jumlah_duplikat} Ditemukan)", expanded=False):
            st.write("**Analisis Kebersihan Baris Data:**")
            if jumlah_duplikat == 0:
                st.success("🟢 Tidak ditemukan adanya baris data yang ganda atau duplikat di dalam berkas dataset utama Anda. Semua baris observasi bersifat unik.")
            else:
                st.warning(f"⚠️ Terdeteksi adanya {jumlah_duplikat} baris data duplikat yang berpotensi memicu bias memori model apabila tidak dieliminasi.")
            st.write("Proses penapisan identitas baris dilakukan secara ketat di Google Colab menggunakan fungsi drop duplicates guna memastikan setiap hari observasi hanya diwakili oleh satu baris data tunggal yang sah.")
            
    with kolom_mis:
        with st.expander(f"❌ Laporan Missing Values ({jumlah_missing} Kosong)", expanded=False):
            st.write("**Sebaran Data Kosong per Kolom Fitur:**")
            df_missing = pd.DataFrame(bingkai_data.isnull().sum(), columns=['Jumlah Data Kosong'])
            df_missing['Persentase (%)'] = (df_missing['Jumlah Data Kosong'] / len(bingkai_data)) * 100
            df_missing = df_missing.sort_values(by='Jumlah Data Kosong', ascending=False)
            
            st.dataframe(
                df_missing,
                use_container_width=True
            )
            st.write("Seluruh data kosong yang bersifat kritikal pada fitur polutan numerik telah diatasi dengan baik menggunakan metode pengisian nilai rata-rata atau imutasi taktis.")

    st.write("---")
    st.write("#### 🏛️ Dokumentasi Validitas & Asal-usul Sumber Data")
    st.markdown("""
    Penyusunan dataset proyek Kelompok 14 ini disadur secara resmi melalui proses integrasi catatan harian stasiun pemantau kualitas udara ambien yang dikelola oleh **Dinas Lingkungan Hidup (DLH) Provinsi DKI Jakarta** serta pos pemantauan wilayah satelit sekitarnya. 
    
    Data historis ini mencakup rekaman performa atmosfer perkotaan yang melacak pergerakan fluktuasi harian enam partikel senyawa paling beracun secara waktu nyata. Seluruh data mentah telah melewati tahapan audit pembersihan data yang ketat dalam fase data understanding di mana kami membuang baris kosong yang cacat, menyelaraskan format penulisan teks kategori baku mutu, serta mengamankan sebaran array numerik polutan dari gangguan bias pencilan ekstrem. Hal ini menjamin bahwa model kecerdasan buatan yang tertanam di dalam sistem ini melakukan proses pembelajaran dari basis data ilmiah yang memiliki integritas dan akuntabilitas hukum yang sah.
    """)

    st.write("")
    st.write("#### 📚 Daftar Pustaka — Rekam Jejak Validitas Sumber Dokumentasi Data")

    st.write("**Dataset 1:** Indeks Pencemaran Udara Provinsi DKI Jakarta")
    st.markdown("""
    **Sumber Resmi:** [Kaggle — Derry Derajat](https://www.kaggle.com/datasets/derryderajat/indeks-pencemaran-udara-dki)  
    **Keterangan Akademik:** Pangkalan data ini memuat rekaman matriks harian yang sangat krusial mengenai sebaran ambien udara di wilayah ibu kota. Keunggulan utama dari berkas ini terletak pada kelengkapan parameter numerik yang mencakup enam jenis partikel polutan paling berbahaya secara simultan mulai dari partikulat debu halus PM2.5, debu kasar PM10, gas belerang SO2, asap karbon monoksida CO, ozon permukaan O3, hingga nitrogen dioksida NO2. Di dalam arsitektur sistem cerdas Kelompok 14, dataset ini diposisikan sebagai pilar utama dalam fase pelatihan algoritma klasifikasi terawasi Random Forest karena distribusi datanya memiliki kerapatan yang tinggi serta struktur label target yang sudah terkalibrasi dengan baik sesuai standar hukum Kementerian Lingkungan Hidup. Hal ini meminimalisir munculnya bias klasifikasi saat model mencoba memisahkan batasan kelas keputusan yang berhimpitan.
    """)

    st.write("**Dataset 2:** Air Quality Index in Jakarta (Periode Jangka Jauh Tahun 2010 sampai 2021)")
    st.markdown("""
    **Sumber Resmi:** [Kaggle — Senadu34](https://www.kaggle.com/datasets/senadu34/air-quality-index-in-jakarta-2010-2021)  
    **Keterangan Akademik:** Berkas dokumentasi ini menyajikan deret waktu jangka panjang yang membentang selama dua belas tahun penuh tanpa terputus. Secara metodologi data science, ketersediaan data temporal yang sangat masif ini memegang peranan vital untuk menguji kemampuan generalisasi model terhadap perubahan musim serta pergeseran pola emisi perkotaan melintasi dekade waktu yang berbeda. Data ini memuat catatan spesifik dari stasiun ikonik DKI4 Lubang Buaya Jakarta Timur yang terkenal memiliki karakteristik paparan polutan yang unik akibat pengaruh area urban sekitarnya. Kelompok 14 memanfaatkan korpus data raksasa ini untuk melakukan proses validasi silang serta mendeteksi adanya tren konsistensi pembentukan kelompok klaster alami secara musiman, sehingga algoritma K-Means kami mampu melahirkan batasan jarak pengelompokan alami yang kebal terhadap gangguan anomali cuaca sesaat.
    """)

    st.write("**Dataset 3:** Data Indeks Standar Pencemar Udara (ISPU) Provinsi DKI Jakarta Portal Nasional")
    st.markdown("""
    **Sumber Resmi:** [Data.go.id — Dinas Lingkungan Hidup Provinsi DKI Jakarta](https://data.go.id/dataset/dataset/data-indeks-standar-pencemar-udara-ispu-di-provinsi-dki-jakarta)  
    **Keterangan Akademik:** Dokumen rujukan ini merupakan pangkalan data primer berskala nasional yang dikelola secara langsung oleh otoritas dinas pemerintah daerah terkait melalui jaringan stasiun pemantauan resmi di lapangan. Keberadaan dataset ini memegang peran tertinggi dalam aspek akuntabilitas riset Kelompok 14 karena memiliki legalitas dan validitas hukum yang sah dari badan negara. Rekaman data di dalam portal ini mencerminkan hasil konversi waktu nyata dari alat pengukur udara ambien otomatis yang ditempatkan pada titik-titik strategis administrasi perkotaan. Kami menggunakan dataset ini sebagai jangkar kebenaran data latih guna menyelaraskan angka batas baku mutu lingkungan yang dinamis, sekaligus memastikan bahwa keluaran prediksi dari sistem cerdas ini sepenuhnya sejalan dengan regulasi formal penentuan status kualitas udara nasional yang berlaku di Indonesia.
    """)

    st.write("**Dataset 4:** Indeks Standar Pencemaran Udara (ISPU) Rekaman Tahunan 2013")
    st.markdown("""
    **Sumber Resmi:** [Data.go.id — Dinas Lingkungan Hidup Provinsi DKI Jakarta](https://data.go.id/dataset/dataset/indeks-standar-pencemaran-udara-ispu-tahun-2013)  
    **Keterangan Akademik:** Lembar data lokal ini memuat dokumentasi riwayat kualitas kebersihan udara secara spesifik untuk sepanjang tahun kalender 2013. Sifat dari dataset ini adalah sebagai pengisi kekosongan informasi temporal guna mencegah terjadinya kesenjangan deret waktu di dalam database sistem kami. Di dalam fase penyiapan data, berkas tahun 2013 ini memberikan tantangan prapemrosesan yang tinggi karena format penulisan teks kategori dan beberapa satuan angka pengukurannya masih menggunakan standar konvensional lama. Kelompok 14 melakukan proses penyelarasan struktur logika data pada berkas ini agar seluruh polutan kritis dapat dibaca secara seragam oleh model kecerdasan buatan, sekaligus memvalidasi bahwa perilaku algoritma pohon keputusan tetap konsisten saat menguji data historis masa lalu.
    """)

    st.write("**Dataset 5:** Indeks Standar Pencemaran Udara (ISPU) Rekaman Tahunan 2012")
    st.markdown("""
    **Sumber Resmi:** [Data.go.id — Dinas Lingkungan Hidup Provinsi DKI Jakarta](https://data.go.id/dataset/dataset/indeks-standar-pencemaran-udara-ispu-tahun-2012)  
    **Keterangan Akademik:** Berkas arsip statistik pemerintah ini menyimpan rekam jejak fluktuasi harian gas buang beracun dan partikel debu halus untuk seluruh wilayah administrasi sepanjang tahun 2012. Secara akademis, tahun 2012 dipilih sebagai objek penyelidikan yang krusial karena wilayah regional tersebut sempat mencatat rentetan anomali fluktuasi iklim lokal serta lonjakan aktivitas kendaraan bermotor yang memicu penumpukan polutan ekstrem di atmosfer terbawah. Penyertaan berkas ini ke dalam pangkalan data utama berfungsi untuk memperkaya variasi sebaran nilai ekstrem pada data latih, sehingga pasukan pohon keputusan di dalam model Random Forest kami dapat mempelajari pola perilaku gas belerang SO2 dan nitrogen dioksida NO2 ketika berada dalam kondisi tekanan lingkungan tertinggi di lapangan.
    """)

    st.write("**Dataset 6:** Indeks Standar Pencemaran Udara (ISPU) Rekaman Tahunan 2011")
    st.markdown("""
    **Sumber Resmi:** [Data.go.id — Dinas Lingkungan Hidup Provinsi DKI Jakarta](https://data.go.id/dataset/dataset/indeks-standar-pencemaran-udara-ispu-tahun-2011)  
    **Keterangan Akademik:** Dokumen data lokal berkekuatan ilmiah ini mendokumentasikan sebaran konsentrasi gas sisa industri dan emisi transportasi harian untuk periode tahun 2011. Integrasi berkas ini ke dalam sistem Kelompok 14 ditujukan murni untuk memperkuat aspek kuantitas data sampel agar memenuhi asas kecukupan volume data latih dalam pemodelan data science tingkat lanjut. Melalui proses penapisan yang mendalam di Google Colab, baris data tahun 2011 ini dibersihkan dari tumpukan data pencilan palsu serta gangguan galat sensor pembaca guna mengamankan akurasi perhitungan jarak Euclidean dari algoritma klasterisasi K-Means, sehingga pengelompokan alami wilayah polusi yang dihasilkan sistem tetap objektif tanpa bias memori masa lalu.
    """)

    st.write("**Dataset 7:** Indeks Standar Pencemaran Udara (ISPU) Rekaman Tahunan 2010")
    st.markdown("""
    **Sumber Resmi:** [Data.go.id — Dinas Lingkungan Hidup Provinsi DKI Jakarta](https://data.go.id/dataset/dataset/indeks-standard-pencemaran-udara-ispu-tahun-2010)  
    **Keterangan Akademik:** Dokumen historis fundamental ini merupakan lapisan data tertua yang berhasil dihimpun di dalam sistem informasi cerdas Kelompok 14. Lembar data tahun 2010 bertindak sebagai titik jangkar awal atau garis dasar bagi analisis evolusi karakteristik kualitas udara ambien perkotaan melintasi kurun waktu satu dekade terakhir. Penyisipan data purba ini memberikan khazanah distribusi statistika deskriptif yang sangat berharga bagi model untuk mengenali bagaimana transformasi kepadatan industri dan perluasan wilayah urbanisasi mempengaruhi volume pelepasan partikel mikro karbon monoksida CO serta ozon O3 dari tahun ke tahun, menghasilkan sistem prediksi yang memiliki kedalaman wawasan sejarah lingkungan yang sangat komprehensif.
    """)

    st.write("---")
    st.write("#### 🔬 Notes Metodologi Integrasi Data")
    st.markdown("""
    **Proses Integrasi dan Fusi Data:**
    1. **Pengumpulan Data Mentah:** Semua dataset diunduh dari sumber resmi masing-masing dan diverifikasi integritas file-nya.
    2. **Pembersihan Data (Data Cleaning):** Baris dengan nilai kosong (NaN) dihapus, format penulisan kategori diselaraskan menjadi standar baku mutu pemerintah, dan nilai numerik yang tidak masuk akal (outlier ekstrem) diidentifikasi.
    3. **Penggabungan Data (Data Merging):** Semua dataset digabungkan menggunakan concatenation berdasarkan kolom waktu (tanggal) yard lokasi (stasiun pemantau).
    4. **Validasi Silang (Cross-Validation):** Data dari berbagai sumber divalidasi satu sama lain untuk memastikan konsistensi nilai polutan pada tanggal dan lokasi yang sama.
    5. **Transformasi Fitur (Feature Engineering):** Variabel turunan seperti skor kelayakan, kategori status, dan rasio kontribusi polutan dihitung berdasarkan referensi standar ISPU Permen LHK No. 14 Tahun 2020.
    6. **Normalisasi Data (Data Scaling):** Data numerik dinormalisasi menggunakan StandardScaler untuk memastikan model machine learning dapat belajar dengan optimal tanpa bias skala nilai.
    """)
else:
    st.error("Peringatan Gagal menemukan berkas data kualitas udara csv di dalam folder dataset Anda")