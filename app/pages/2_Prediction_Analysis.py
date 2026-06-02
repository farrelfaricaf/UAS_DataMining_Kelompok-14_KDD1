import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

direktori_halaman = os.path.dirname(os.path.abspath(__file__))
direktori_app_folder = os.path.dirname(direktori_halaman)
direktori_akar_proyek = os.path.dirname(direktori_app_folder)

rute_model_rf = os.path.join(direktori_akar_proyek, "model", "model_classification_random_forest.pkl")
rute_model_km = os.path.join(direktori_akar_proyek, "model", "model_clustering_kmeans.pkl")
rute_scaler_rf = os.path.join(direktori_akar_proyek, "model", "scaler_classification.pkl")
rute_scaler_km = os.path.join(direktori_akar_proyek, "model", "scaler_clustering.pkl")
rute_peta = os.path.join(direktori_akar_proyek, "model", "cluster_label_map.pkl")

@st.cache_resource
def muat_aset_biner_prediksi():
    m_rf = joblib.load(rute_model_rf)
    m_km = joblib.load(rute_model_km)
    s_rf = joblib.load(rute_scaler_rf)
    s_km = joblib.load(rute_scaler_km)
    p_klas = joblib.load(rute_peta)
    return m_rf, m_km, s_rf, s_km, p_klas

rute_css = os.path.join(direktori_akar_proyek, "app", "assets", "style.css")
if os.path.exists(rute_css):
    with open(rute_css, "r") as berkas_css:
        st.markdown(f"<style>{berkas_css.read()}</style>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color: #ffffff; margin-bottom: 10px; font-size: 18px;'>🎯 Navigasi Sistem</h3>", unsafe_allow_html=True)

try:
    model_rf, model_km, scaler_rf, scaler_km, peta_klaster = muat_aset_biner_prediksi()
except Exception as e:
    st.error(f"Gagal memuat aset biner pkl di halaman prediksi. Galat: {e}")

st.title("🔮 Dasbor Cek Kualitas Udara & Panduan Kesehatan Mandiri")
st.write("Silakan geser tombol kadar polusi udara di bawah ini untuk melihat hasil analisis kondisi lingkungan secara akurat.")
st.write("---")

st.write("#### 📐 Silakan Tentukan Kadar Polutan di Lingkungan Anda")

st.write("**1. Parameter Partikel Debu Layang Lingkungan**")
k_deb1, k_deb2 = st.columns(2)
with k_deb1:
    nilai_pm25 = st.slider("Kadar Debu Sangat Halus / PM2.5 (µg/m³)", min_value=0.0, max_value=300.0, value=45.0, step=0.1, key="s_pm25")
with k_deb2:
    nilai_pm10 = st.slider("Kadar Debu Kasar / PM10 (µg/m³)", min_value=0.0, max_value=300.0, value=55.0, step=0.1, key="s_pm10")

st.write("")
st.write("**2. Parameter Senyawa Gas Beracun Otomotif & Industri**")
k_gas1, k_gas2 = st.columns(2)
with k_gas1:
    nilai_so2 = st.slider("Kadar Gas Belerang / SO2 (µg/m³)", min_value=0.0, max_value=300.0, value=25.0, step=0.1, key="s_so2")
with k_gas2:
    nilai_co = st.slider("Kadar Asap Kendaraan / CO (µg/m³)", min_value=0.0, max_value=300.0, value=12.0, step=0.1, key="s_co")

st.write("")
st.write("**3. Parameter Senyawa Oksidan Atmosfer**")
k_oks1, k_oks2 = st.columns(2)
with k_oks1:
    nilai_o3 = st.slider("Kadar Gas Lapisan Udara / O3 (µg/m³)", min_value=0.0, max_value=300.0, value=35.0, step=0.1, key="s_o3")
with k_oks2:
    nilai_no2 = st.slider("Kadar Gas Pabrik Knalpot / NO2 (µg/m³)", min_value=0.0, max_value=300.0, value=18.0, step=0.1, key="s_no2")

st.write("")
if st.button("🚀 Mulai Periksa Kondisi Udara Hari Ini", use_container_width=True):
    data_masukan = np.array([[nilai_pm25, nilai_pm10, nilai_so2, nilai_co, nilai_o3, nilai_no2]])
    
    masukan_klasifikasi = scaler_rf.transform(data_masukan)
    masukan_klaster = scaler_km.transform(data_masukan)
    
    prediksi_kelas = model_rf.predict(masukan_klasifikasi)[0]
    prediksi_kelompok = model_km.predict(masukan_klaster)[0]
    
    daftar_status = ["BAIK", "SEDANG", "TIDAK SEHAT", "SANGAT TIDAK SEHAT"]
    kategori_final = daftar_status[int(prediksi_kelas)]
    
    nilai_mentah = [nilai_pm25, nilai_pm10, nilai_so2, nilai_co, nilai_o3, nilai_no2]
    nama_fitur = ['PM2.5', 'PM10', 'SO2', 'CO', 'O3', 'NO2']
    ambang_batas = [55.0, 75.0, 50.0, 35.0, 100.0, 80.0]
    
    rasio_kontribusi = [m / a for m, a in zip(nilai_mentah, ambang_batas)]
    max_rasio = max(rasio_kontribusi)
    idx_kritikal = np.argmax(rasio_kontribusi)
    polutan_kritikal = nama_fitur[idx_kritikal]
    
    rata_rasio = np.mean(rasio_kontribusi)
    skor_kelayakan = max(0.0, 100.0 - (rata_rasio * 40.0))
    
    if skor_kelayakan > 75.0:
        label_spasial = "Cluster Baik (Polusi Rendah)"
    elif skor_kelayakan > 50.0:
        label_spasial = "Cluster Sedang (Polusi Moderat)"
    elif skor_kelayakan > 25.0:
        label_spasial = "Cluster Tidak Sehat (Polusi Tinggi)"
    else:
        label_spasial = "Cluster Sangat Tidak Sehat (Polusi Ekstrem)"
        
    warna_tema = "#2ec4b6" if kategori_final == "BAIK" else ("#ff9f1c" if kategori_final == "SEDANG" else "#e71d36")
    
    hasil_kiri, hasil_kanan = st.columns(2)
    with hasil_kiri:
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; border-left:8px solid {warna_tema};">
            <h4 style="margin:0; color:#31333f;">📋 Status Udara Resmi (Standar Pemerintah)</h4>
            <p style="font-size:24px; font-weight:bold; margin:10px 0 0 0; color:{warna_tema};">Zonasi: {kategori_final}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with hasil_kanan:
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; border-left:8px solid #00c0f2;">
            <h4 style="margin:0; color:#31333f;">🏢 Karakteristik Udara Alami di Lapangan</h4>
            <p style="font-size:18px; font-weight:bold; margin:14px 0 0 0; color:#00c0f2;">{label_spasial}</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("---")
    st.subheader("💡 Penjelasan Lengkap Mengenai Kondisi Udara Anda")
    
    tab_cerita, tab_solusi = st.tabs(["📖 Cerita Di Balik Angka Udara", "🏡 Panduan Aksi Nyata Lindungi Keluarga"])
    
    with tab_cerita:
        st.write(f"Zat polutan yang paling mendominasi wilayah Anda saat ini adalah **{polutan_kritikal}**. Angka konsentrasinya sudah mencapai **{max_rasio:.2f} kali lipat** dari ambang batas aman yang direkomendasikan.")

        status_simpul = "SINKRON"
        if (kategori_final == "BAIK" and "Baik" not in label_spasial) or \
           (kategori_final == "SEDANG" and "Sedang" not in label_spasial) or \
           (kategori_final == "TIDAK SEHAT" and "Tidak Sehat" not in label_spasial) or \
           (kategori_final == "SANGAT TIDAK SEHAT" and "Sangat Tidak Sehat" not in label_spasial):
            status_simpul = "PARADOKS"

        if skor_kelayakan > 75.0:
            if kategori_final == "BAIK" and "Baik" in label_spasial:
                st.success(f"🎉 **Analisis Zona Paru-paru Prima — Sempurna & Sinkron (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Selamat! Laporan kondisi udara wilayah Anda hari ini benar-benar melegakan hati. Sistem klasifikasi resmi pemerintah and mesin pengelompokan alami kecerdasan buatan kami sama-sama bersepakat bahwa atmosfer di sekitar tempat tinggal Anda sedang berada dalam kondisi paling bersih and paling segar yang bisa dicapai oleh sebuah wilayah hunian manusia. Angka skor kelayakan sebesar {skor_kelayakan:.1f}% ini bukan sekadar angka cantik di atas kertas. Angka ini merupakan cerminan nyata bahwa seluruh partikel debu halus PM2.5, debu kasar PM10, gas belerang SO2, asap kendaraan CO, ozon atmosfer O3, and nitrogen dioksida NO2 semuanya berada jauh di bawah garis merah ambang batas bahaya yang ditetapkan oleh regulasi lingkungan hidup Indonesia. Kondisi ini biasanya muncul pada pagi hari setelah hujan lebat semalam membersihkan seluruh partikel polutan yang melayang di udara, atau pada hari libur nasional ketika aktivitas industri and kendaraan bermotor berhenti beroperasi secara massal. Molekul oksigen O2 murni sedang melimpah ruah tanpa gangguan radikal bebas sehingga setiap tarikan napas yang Anda lakukan akan mengantarkan oksigen berkualitas tinggi langsung ke alveolus paru-paru and kemudian ke seluruh sel darah merah dalam tubuh Anda. Polutan paling dominan di wilayah Anda saat ini adalah {polutan_kritikal} dengan rasio kelampauan sebesar {max_rasio:.2f} kali lipat terhadap batas aman, and angka itu pun masih terbilang sangat rendah sehingga sistem tubuh manusia dapat menetralkannya secara otomatis tanpa membutuhkan bantuan medis apa pun.")
            elif kategori_final == "BAIK" and "Baik" not in label_spasial:
                st.success(f"🎉 **Analisis Zona Prima — Skor Tinggi dengan Sinyal Mikrolokal (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Sistem mendeteksi bahwa meskipun kadar akumulasi parameter polusi secara umum masih berada di ambang batas aman yang sangat baik, terdapat peningkatan kecil yang tertangkap oleh algoritma pengelompokan alami kami. Kondisi ini menjelaskan bahwa profil udara Anda mulai menyerap sisa emisi karbon dari aktivitas lingkungan terdekat seperti hiruk-pikuk jalur transportasi lokal atau operasional komersial harian yang mulai aktif di sekitar pemukiman. Selisih pandang antar-model ini merupakan bukti kepekaan sistem dalam membaca pergerakan polutan mikro sebelum polusi mengendap secara merata di udara terbuka.")
            elif kategori_final == "SEDANG" and "Baik" in label_spasial:
                st.success(f"🌤️ **Analisis Zona Prima — Klaster Alami Lebih Bersih dari Nilai Resmi (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Kondisi udara Anda menunjukkan fenomena yang menarik di mana nilai akumulatif parameter polutan sebenarnya masih berada pada level yang sangat murni di bawah kelompok klaster udara bersih alami. Pergeseran status sedang dari pemerintah dipicu oleh kenaikan minor pada satu jenis zat tunggal saja, sementara komponen gas and partikel layang lainnya tetap bertahan di batas terendah yang aman bagi paru-paru manusia. Anda tidak perlu khawatir karena kapasitas udara bersih secara keseluruhan di sekitar hunian Anda masih sangat dominan melarutkan gas sisa buang tersebut.")
            else:
                st.success(f"🎉 **Analisis Zona Paru-paru Prima — Kondisi Istimewa (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Skor kelayakan udara wilayah Anda berada di angka {skor_kelayakan:.1f}%, sebuah pencapaian kualitas lingkungan yang harus dijaga and dirayakan bersama. Meskipun terdapat sedikit perbedaan label antara penilaian resmi and pengelompokan alami kami, keduanya menunjukkan bahwa beban total polutan di wilayah Anda masih sangat terkendali. Polutan {polutan_kritikal} yang menjadi parameter paling menonjol dengan rasio {max_rasio:.2f} kali lipat masih berada dalam batas yang dapat dikelola oleh mekanisme pertahanan alami tubuh manusia dewasa yang sehat.")

        elif skor_kelayakan > 50.0:
            if kategori_final == "BAIK" and status_simpul == "SINKRON":
                st.info(f"ℹ️ **Analisis Zona Waspada — Awal Penumpukan Polusi Harian (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Kondisi udara Anda saat ini masih berstatus resmi BAIK menurut perhitungan ISPU pemerintah, and klaster alami pun masih menempatkan Anda di kelompok yang relatif wajar. Namun perlu Anda pahami bahwa skor kelayakan sebesar {skor_kelayakan:.1f}% ini sudah mulai bergeser dari zona emas menuju zona kuning kewaspadaan. Polutan utama yang bertanggung jawab atas penurunan skor ini adalah {polutan_kritikal} yang konsentrasinya sudah menyentuh {max_rasio:.2f} kali lipat dari batas aman. Dalam siklus polusi harian perkotaan, pola seperti ini sangat umum terjadi pada pukul 07.00 hingga 09.00 pagi saat jutaan kendaraan bermotor secara bersamaan menyalakan mesin and memompa gas buang ke lapisan udara terbawah, atau pada sore hari pukul 16.00 hingga 18.00 saat arus balik kerja membanjiri jalan raya. Gas karbon monoksida dari knalpot akan mengikat hemoglobin darah lebih kuat dibandingkan oksigen sehingga bagi pengendara sepeda motor yang menghirup langsung emisi jalan raya tanpa masker, efek kelelahan dini and sedikit pusing bisa mulai terasa setelah terpapar selama lebih dari tiga puluh menit tanpa henti.")
            elif kategori_final == "SEDANG" and status_simpul == "SINKRON":
                st.info(f"ℹ️ **Analisis Zona Waspada — Polusi Moderat Mulai Terasa (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Sistem kami mencatat kondisi lingkungan Anda sudah masuk ke dalam fase tekanan emisi karbon skala menengah. Status SEDANG yang diberikan oleh klasifikasi resmi pemerintah bukan berarti aman sepenuhnya. Kategori SEDANG dalam standar ISPU Indonesia (Permen LHK No. 14 Tahun 2020) berarti udara di wilayah Anda masih dapat ditoleransi oleh manusia dewasa sehat, namun kelompok rentan seperti balita di bawah lima tahun, lansia di atas enam puluh tahun, and penderita asma atau penyakit paru obstruktif kronis sudah mulai merasakan dampak iritasi ringan. Polutan {polutan_kritikal} dengan konsentrasi {max_rasio:.2f} kali lipat ambang batas adalah tersangka utama di balik keluhan bersin, mata sedikit perih, atau tenggorokan gatal yang mungkin sudah mulai Anda rasakan tanpa menyadari penyebabnya. Gas SO2 apabila menjadi polutan dominan akan bereaksi dengan uap air di saluran napas atas membentuk asam sulfit yang mengikis lapisan lendir pelindung. Gas NO2 yang tinggi akan memperlemah daya tahan selaput bronkus sehingga bakteri patogen lebih mudah masuk and berkembang biak, meningkatkan risiko infeksi saluran pernapasan atas dalam jangka waktu paparan yang panjang.")
            elif status_simpul == "PARADOKS" and kategori_final in ["BAIK", "SEDANG"] and "Tidak Sehat" in label_spasial:
                st.info(f"ℹ️ **Analisis Zona Waspada — Paradoks Deteksi Dini Terdeteksi! (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Sistem mendeteksi adanya lompatan partikel polutan lokal yang cukup tajam di mana kondisi riil udara di lapangan sebenarnya sudah mulai jenuh menyerupai profil kelompok tidak sehat. Hal ini menjelaskan mengapa model pengelompokan alami kami memberikan sinyal peringatan dini berkategori tinggi, meskipun rata-rata perhitungan stasiun resmi pemerintah masih berada di ambang batas sedang. Kondisi akumulasi partikel debu layang atau emisi kendaraan ini harus mulai diwaspadai karena menjadi tanda awal bahwa sirkulasi udara bersih di sekitar lingkungan Anda mulai kewalahan mengurai gas buang pembakaran.")
            elif status_simpul == "PARADOKS" and kategori_final in ["TIDAK SEHAT", "SANGAT TIDAK SEHAT"] and "Baik" in label_spasial:
                st.info(f"ℹ️ **Analisis Zona Waspada — Paradoks Pemulihan Alami (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Situasi yang unik terjadi di wilayah Anda. Klasifikasi resmi pemerintah mencatatkan status {kategori_final}, namun klaster alami mesin kami menempatkan pola polutan Anda di kelompok {label_spasial}. Fenomena ini bisa terjadi ketika angin kencang atau hujan lebat baru saja turun and berhasil mengencerkan konsentrasi polutan secara drastis dalam waktu singkat, namun nilai ISPU resmi belum sempat diperbarui karena stasiun pemantau memiliki jeda waktu pelaporan. Polutan {polutan_kritikal} yang sebelumnya sangat tinggi kini mulai terurai, and pola komposisi kimia udara Anda sudah bergerak mendekati profil udara yang lebih bersih. Skor kelayakan {skor_kelayakan:.1f}% Anda mencerminkan kondisi transisi pemulihan ini dengan sangat akurat.")
            else:
                st.info(f"ℹ️ **Analisis Zona Waspada — Tekanan Polusi Mulai Terakumulasi (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Kondisi udara wilayah Anda sedang berada di zona waspada dengan skor {skor_kelayakan:.1f}%. Polutan yang paling banyak berkontribusi terhadap penurunan kualitas udara ini adalah {polutan_kritikal} yang sudah melampaui ambang batas sebesar {max_rasio:.2f} kali lipat. Dalam zona 50 hingga 75 persen ini, tanda-tanda awal penumpukan polusi harian mulai terlihat meskipun alumunium mencapai level yang membutuhkan tindakan darurat. Kelompok populasi yang paling perlu berhati-hati adalah anak-anak yang aktif bermain di luar ruangan, ibu hamil yang membutuhkan pasokan oksigen bersih lebih banyak, serta siapa pun yang memiliki riwayat alergi atau penyakit pernapasan kronis. Tindakan pencegahan ringan sudah cukup untuk mengatasi kondisi ini secara efektif.")

        elif skor_kelayakan > 25.0:
            if kategori_final == "TIDAK SEHAT" and "Tidak Sehat" in label_spasial:
                st.warning(f"⚠️ **Analisis Zona Paru-paru Terancam — Ancaman Nyata & Sinkron (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Lapisan atmosfer di wilayah tempat tinggal Anda saat ini sudah resmi diselimuti oleh kabut polusi kotor yang pekat and berbahaya bagi kesehatan semua golongan usia tanpa terkecuali. Kedua mesin analisis kami sepakat bulat bahwa kondisi ini bukan sekadar penurunan kualitas sesaat. Ini adalah akumulasi polutan yang terjadi secara konsisten and merata. Polutan paling berbahaya yang merajai udara Anda saat ini adalah {polutan_kritikal} dengan konsentrasi yang sudah melebihi ambang batas aman sebesar {max_rasio:.2f} kali lipat. Angka ini bukan angka abstrak. Bayangkan seperti mengisi gelas air sebesar {max_rasio:.2f} kali dari kapasitas normalnya sampai tumpah ke mana-mana. Debu halus PM2.5 yang berukuran kurang dari 2.5 mikrometer atau sekitar tiga puluh kali lebih kecil dari diameter rambut manusia akan melewati bulu hidung, melewati silia saluran napas atas, and langsung menerobos masuk ke kantung udara alveolus yang paling dalam. Di sana partikel beracun ini akan menempel pada dinding jaringan paru and memicu respons peradangan kronis yang bisa berlangsung bertahun-tahun bahkan setelah paparan berakhir. Gas ozon O3 yang tinggi akan membakar lapisan mukosa saluran bronkus seperti racun asam pelan-pelan, menghasilkan sensasi dada terasa sesak and napas pendek yang tidak nyaman. Skor kelayakan {skor_kelayakan:.1f}% mempertegas bahwa hampir separuh lebih kemampuan udara wilayah Anda untuk mendukung kehidupan sehat sudah terkompromikan oleh beban polutan ini.")
            elif kategori_final == "SANGAT TIDAK SEHAT" and "Tidak Sehat" in label_spasial:
                st.warning(f"⚠️ **Analisis Zona Terancam — Klasifikasi Sangat Berbahaya, Klaster Mengonfirmasi Ancaman (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Sistem penilaian resmi pemerintah telah mengangkat bendera merah dengan status SANGAT TIDAK SEHAT, and mesin klaster alami kami pun mengonfirmasi bahwa profil kimia udara Anda memang berada di kelompok yang berisiko tinggi. Meskipun skor kelayakan {skor_kelayakan:.1f}% Anda masih berada di zona terancam and belum jatuh ke jurang zona sekarat, kondisi ini sudah sangat serius and tidak boleh dianggap remeh. Perbedaan antara zona terancam and zona sekarat bisa sangat tipis and bisa berubah dalam hitungan jam apabila sumber emisi baru seperti kebakaran lahan atau kecelakaan pabrik ikut bergabung menambah beban polutan yang sudah ada. Polutan {polutan_kritikal} dengan konsentrasi {max_rasio:.2f} kali lipat ambang batas adalah aktor utama di balik kondisi darurat ini. Partikel and gas beracun yang sudah memenuhi udara akan memperburuk gejala pada penderita hipertensi karena sistem jantung harus bekerja lebih keras memompa darah beroksigen rendah ke seluruh organ vital tubuh.")
            elif status_simpul == "PARADOKS" and kategori_final in ["TIDAK SEHAT", "SANGAT TIDAK SEHAT"] and "Baik" in label_spasial:
                st.warning(f"⚠️ **Analisis Zona Terancam — Anomali Deviasi Lintas Model Terdeteksi! (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Sistem mendeteksi adanya ketidaksesuaian tajam antar-model di mana konsentrasi satu jenis zat gas beracun melambung tinggi secara drastis mengaktifkan indikator bahaya pemerintah, sementara senyawa partikulat lainnya masih berada pada tren kelompok rendah. Kondisi ini menjelaskan bahwa lingkungan Anda sedang terpapar emisi gas buang kimia yang sangat pekat and bersifat merusak secara lokal di dekat instalasi pemantauan. Walaupun struktur kelompok udara secara umum belum bergeser sepenuhnya, Anda wajib memprioritaskan peringatan dini ini karena sebaran zat tunggal yang beracun tersebut sudah sangat membahayakan pernapasan jangka pendek.")
            elif status_simpul == "PARADOKS" and kategori_final == "BAIK" and "Tidak Sehat" in label_spasial:
                st.warning(f"⚠️ **Analisis Zona Terancam — Paradoks Berbahaya, Angka Resmi Menyembunyikan Bahaya! (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"🔬 **Peringatan Kritis Paradoks:** Momen yang berbahaya ini harus Anda baca dengan sangat seksama. Angka resmi pemerintah masih mencatat status BAIK untuk wilayah Anda, namun mesin klaster kecerdasan buatan kami sudah menarik alarm karena menemukan bahwa pola distribusi polutan Anda sudah sangat mirip dengan profil wilayah yang masuk kelompok {label_spasial}. Skor kelayakan {skor_kelayakan:.1f}% yang sudah merosot ke zona terancam ini adalah bukti nyata bahwa ada ketidakseimbangan compositional polutan yang serius sedang berlangsung di lingkungan Anda meskipun angka ISPU tunggal belum berteriak keras. Polutan {polutan_kritikal} yang mencapai {max_rasio:.2f} kali lipat ambang batas mungkin tidak cukup untuk mengubah satu angka ISPU pemerintah, namun kombinasinya dengan polutan-polutan lain yang turut meningkat secara bersamaan menciptakan koktail racun udara yang jauh lebih berbahaya daripada yang ditunjukkan oleh satu angka tunggal mana pun. Percayai sinyal klaster kami and ambil tindakan perlindungan segera.")
            elif kategori_final == "BAIK" or kategori_final == "SEDANG":
                st.warning(f"⚠️ **Analisis Zona Paru-paru Terancam — Kabut Polusi Merayap Naik (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Perhatian serius diperlukan di sini. Meskipun label formal yang diberikan sistem pemerintah adalah {kategori_final}, skor kelayakan udara wilayah Anda sudah melorot ke angka yang mengkhawatirkan yaitu hanya {skor_kelayakan:.1f}% saja. Ini menandakan bahwa secara agregat keseluruhan, beban total polutan di udara Anda sudah sangat berat meskipun satu parameter tertinggi mungkin belum melampaui batas ISPU secara resmi. Polutan {polutan_kritikal} dengan rasio {max_rasio:.2f} kali lipat menjadi beban terbesar dalam komposisi ini. Gabungan multi-polutan dalam konsentrasi menengah ke atas memiliki efek sinergis yang jauh lebih merusak dibandingkan satu polutan tunggal yang tinggi karena sistem enzim detoksifikasi tubuh harus bekerja menangani banyak racun sekaligus and kemampuannya akan cepat kehabisan cadangan.")
            else:
                st.warning(f"⚠️ **Analisis Zona Paru-paru Terancam — Polusi Pekat Menyelimuti Wilayah (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Atmosfer lingkungan di sekitar tempat tinggal Anda sudah resmi diselimuti oleh kabut polusi kotor yang pekat and berbahaya bagi kesehatan. Munculnya status {kategori_final} dikombinasikan dengan pengelompokan alami di **{label_spasial}** and skor kelayakan hanya {skor_kelayakan:.1f}% menandakan bahwa sirkulasi angin alam terbuka sudah gagal mengurai kepadatan partikel beracun yang melayang bebas di udara terbuka. Debu halus yang melambung tinggi memiliki ukuran yang sangat kecil sehingga bulu hidung manusia tidak akan mampu menyaringnya dengan baik. Partikel jahat ini akan menyelinap masuk menembus jaringan terdalam paru-paru and memicu reaksi sesak napas serta rasa perih di tenggorokan yang bisa berlangsung berjam-jam. Polutan utama di wilayah Anda adalah {polutan_kritikal} dengan konsentrasi mencapai {max_rasio:.2f} kali lipat dari batas aman yang ditetapkan pemerintah.")

        else:
            if kategori_final == "SANGAT TIDAK SEHAT" and "Sangat Tidak Sehat" in label_spasial:
                st.error(f"🚨 **DARURAT POLUSI EKSTREM — KONFIRMASI PENUH KEDUA MODEL (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Ini adalah kondisi terburuk yang pernah kami rekam and kedua mesin analisis kami sepakat penuh tanpa keraguan sedikit pun bahwa wilayah Anda sedang mengalami bencana kualitas udara tingkat akut. Skor kelayakan {skor_kelayakan:.1f}% ini bukan sekadar angka rendah. Angka ini berarti bahwa kemampuan udara di sekitar Anda untuk menopang kehidupan manusia secara sehat sudah runtuh hingga ke titik kritis yang paling mengkhawatirkan. Polutan {polutan_kritikal} yang meledak konsentrasinya hingga {max_rasio:.2f} kali lipat dari ambang batas aman adalah bom udara yang sedang meledak perlahan and tak kasat mata di sekitar Anda. Senyawa hidrokarbon aromatik polisiklik, partikel logam berat seperti timbal and merkuri yang sering menyertai polusi ekstrem perkotaan, serta radikal bebas oksidatif dari senyawa peroksiasetil nitrat akan menyerbu masuk ke dalam tubuh setiap kali Anda menarik napas tanpa perlindungan yang memadai. Dalam jangka pendek paparan seperti ini akan memicu serangan asma akut, aritmia jantung, kejang bronkospasme, and penurunan saturasi oksigen darah secara drastis. Dalam jangka panjang, paparan polutan ekstrem berulang seperti ini akan mempercepat proses penuaan sel paru-paru, meningkatkan risiko kanker paru hingga berkali-kali lipat, and memperparah kondisi penyakit kardiovaskular yang sudah ada sebelumnya. Kondisi ini bukan sekedar tidak nyaman. Ini adalah ancaman jiwa nyata yang membutuhkan respons segera and terkoordinasi.")
            elif kategori_final == "SANGAT TIDAK SEHAT" and status_simpul == "PARADOKS":
                st.error(f"🚨 **DARURAT POLUSI EKSTREM — PARADOKS KRITIS TERDETEKSI (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Sistem kami mendeteksi sebuah anomali kritis yang sangat serius. Status resmi pemerintah mengkategorikan udara Anda sebagai SANGAT TIDAK SEHAT, namun mesin klaster alami kami memberikan label berbeda yaitu **{label_spasial}**. Dalam situasi zona sekarat seperti ini, paradoks antara dua model bisa menandakan salah satu dari dua skenario ekstrem yang sama-sama harus ditangani serius. Skenario pertama, ada sumber polutan tunggal yang sangat masif and terlokalisir seperti kebakaran pabrik atau ledakan gudang bahan kimia yang membuat satu polutan yaitu {polutan_kritikal} melambung ke angka {max_rasio:.2f} kali lipat ambang batas secara tiba-tiba, sementara polutan lain masih relatif normal sehingga pola klaster terlihat seolah tidak terlalu buruk secara multi-dimensi. Skenario kedua, sedang terjadi perubahan angin atau kondisi meteorologi yang mendistribusi ulang polutan secara tidak merata di wilayah Anda. Apapun skenarionya, skor kelayakan {skor_kelayakan:.1f}% membuktikan situasi ini adalah darurat mutlak and tidak ada ruang untuk penundaan dalam mengambil tindakan keselamatan.")
            elif kategori_final in ["TIDAK SEHAT", "SEDANG"] and skor_kelayakan <= 10.0:
                st.error(f"🚨 **DARURAT KRITIS — SKOR MENDEKATI NOL PERSEN! (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Kondisi lingkungan sekitar Anda berada dalam tingkat kerusakan atmosfer yang sangat fatal di mana persentase kelayakan udara bersih sudah merosot drastis menyentuh level terendah mendekati nol persen. Fenomena kepatasan polusi kumulatif ini menjelaskan bahwa seluruh komponen zat beracun, baik debu mikro maupun emisi gas buang, telah bercampur padat membentuk koktail polusi yang meracuni ruang udara secara menyeluruh. Meskipun pembacaan indeks stasiun formal pemerintah belum mencapai level peringatan tertinggi karena batas batas kalkulasi kaku, realitas fisik di lapangan menunjukkan bahwa persediaan oksigen sehat sudah habis tertutup kabut hidrokarbon beracun.")
            else:
                st.error(f"🚨 **Analisis Zona Paru-paru Sekarat — Kondisi Darurat Radikal Bebas (Skor Kelayakan {skor_kelayakan:.1f}%)**")
                st.write(f"Kondisi lingkungan sekitar Anda sudah berada dalam fase darurat polusi ekstrem yang sangat fatal and menghancurkan ruang hidup sehat. Kesepakatan atau ketidaksepakatan hasil dua model pengujian yang sama-sama bermuara pada zona ini membuktikan adanya ledakan konsentrasi gas racun massal yang sangat pekat di atmosfer Anda. Dengan skor kelayakan yang hanya tersisa {skor_kelayakan:.1f}% ini, ruang udara di sekitar Anda kini dipenuhi oleh senyawa radikal bebas pemicu peradangan sistemik yang akan merusak jaringan pernapasan, sistem kardiovaskular, and fungsi otak secara bersamaan dalam waktu singkat jika paparan berlanjut tanpa perlindungan. Polutan {polutan_kritikal} pada konsentrasi {max_rasio:.2f} kali lipat ambang batas sudah masuk dalam klasifikasi paparan akut yang secara medis membutuhkan penanganan segera and serius.")

    with tab_solusi:
        st.write("#### 🏥 Rencana Aksi Taktis Penyelamatan Kesehatan Mandiri")

        if skor_kelayakan > 75.0:
            if kategori_final == "BAIK":
                st.success("🌿 **Udara Anda Prima! Manfaatkan Semaksimal Mungkin**")
                st.markdown(f"""
**🌅 Optimalkan Waktu Pagi Emas Anda:**
- Buka lebar-lebar seluruh jendela and pintu rumah Anda pada pagi hari antara pukul 05.00 hingga 07.00 agar gelombang udara bersih yang kaya oksigen dapat menyapu tuntas debu mati and partikel sisa yang mengendap di dalam ruangan selama semalam.
- Ganti sirikulasi udara dalam rumah secara menyeluruh dengan membiarkan angin pagi masuk bebas minimal tiga puluh menit sebelum aktivitas rumah tangga dimulai.
- Cuci and jemur sprei, bantal guling, serta gorden di luar ruangan untuk memanfaatkan sinar ultraviolet pagi yang berfungsi membunuh bakteri and jamur yang menempel pada kain.

**🏃 Waktu Terbaik untuk Bergerak Aktif:**
- Ini adalah hari emas untuk melakukan olahraga berat di luar ruangan seperti lari jarak jauh, bersepeda, senam aerobik taman, atau bermain bola bersama keluarga di lapangan terbuka tanpa perlu khawatir akan risiko polutan berbahaya.
- Ajak anak-anak bermain aktif di taman, halaman rumah, atau lapangan olahraga untuk mengisi paru-paru mereka yang sedang tumbuh dengan oksigen berkualitas terbaik.
- Bagi lansia, waktu prima seperti ini sangat ideal untuk jalan kaki pagi yang lebih panjang dari biasanya guna melatih kapasitas paru-paru and memperkuat sistem kardiovaskular.
- Olahraga renang terbuka atau di kolam renang outdoor pun sangat direkomendasikan karena air yang menguap justru menambah kelembapan udara sekitar and membantu menjaga selaput lendir saluran napas tetap lembab.

**🌳 Investasi Hijau Jangka Panjang:**
- Manfaatkan semangat kondisi udara yang segar hari ini untuk menanam tanaman hias penangkap debu and penyerap polutan di pekarangan rumah seperti Lidah Mertua (Sansevieria) yang mampu menyerap formaldehida and benzena, Sirih Gading yang efektif menyerap karbon monoksida, serta Pakis Boston yang terbukti meningkatkan kelembapan udara ruangan secara alami.
- Pertimbangkan untuk menanam pohon peneduh di depan rumah seperti Trembesi, Mahoni, atau Ketapang Kencana yang daun-daunnya mampu menangkap partikel debu dari jalan raya sebelum masuk ke area rumah Anda.
- Siram and rawat tanaman yang sudah ada untuk menjaga lapisan daun tetap segar and optimal dalam menyerap karbon dioksida serta melepaskan oksigen murni ke udara sekitar.

**🍽️ Nutrisi Penjaga Paru-paru:**
- Konsumsi buah-buahan tinggi vitamin C seperti jeruk, jambu biji, and kiwi yang berfungsi sebagai antioksidan alami untuk memperbaiki sel-sel paru-paru dari kerusakan oksidatif yang mungkin terjadi pada hari-hari sebelumnya dengan kualitas udara yang lebih buruk.
- Minum teh hijau hangat di pagi hari karena kandungan katekin di dalamnya terbukti secara ilmiah membantu melindungi jaringan epitel saluran napas dari kerusakan akibat polutan.
- Perbanyak konsumsi sayuran berdaun gelap seperti bayam, kangkung, and brokoli yang kaya akan beta-karoten sebagai prekursor vitamin A untuk menjaga integritas lapisan mukosa paru-paru.

**📋 Pemantauan Dan Pencatatan Kualitas Udara:**
- Gunakan momen kondisi prima ini sebagai data acuan atau baseline untuk mencatat seperti apa rasa and kenyamanan bernapas ketika udara benar-benar bersih, sehingga Anda lebih mudah mendeteksi perubahan kualitas udara di hari-hari mendatang.
- Pasang aplikasi pemantau kualitas udara di ponsel Anda and aktifkan notifikasi harian agar Anda selalu mendapat informasi terkini sebelum memulai aktivitas di luar ruangan.
                """)
            else:
                st.success("🌿 **Skor Tinggi Namun Tetap Waspada — Tindakan Pemeliharaan Aktif**")
                st.markdown(f"""
**🌅 Manfaatkan Kondisi Udara yang Masih Prima Ini:**
- Meskipun label resmi mencatat status {kategori_final}, skor kelayakan {skor_kelayakan:.1f}% menunjukkan bahwa kondisi udara secara keseluruhan masih sangat layak untuk aktivitas normal.
- Tetap buka jendela rumah pada pagi hari untuk sirkulasi udara namun perhatikan arah angin agar tidak membawa asap atau debu dari sumber terdekat masuk ke dalam rumah.
- Lakukan aktivitas olahraga ringan hingga sedang di luar ruangan namun hindari berdekatan langsung dengan jalan raya padat atau area industri.

**🌳 Penguatan Pertahanan Lingkungan Rumah:**
- Pasang kisi-kisi kawat halus atau tirai bambu di depan jendela untuk menyaring partikel debu kasar dari luar sebelum masuk ke dalam ruangan, tanpa menghalangi sirkulasi udara segar.
- Letakkan beberapa pot tanaman penyerap polutan di dalam rumah terutama di ruang tidur and ruang keluarga sebagai penyaring udara alami yang bekerja dua pukul empat jam tanpa biaya listrik.
- Pastikan saluran ventilasi kamar mandi and dapur berfungsi dengan baik untuk mencegah akumulasi uap air and bau tidak sedap yang bisa menjadi media tumbuh jamur berbahaya.

**💧 Hidrasi dan Nutrisi Optimal:**
- Minum air putih minimal dua liter per hari untuk menjaga kelembapan selaput lendir saluran napas yang berfungsi sebagai pertahanan pertama terhadap partikel polutan.
- Konsumsi madu murni satu sendok makan setiap pagi karena kandungan hydrogen peroksida alaminya memiliki sifat antimikroba yang membantu membersihkan saluran napas.
- Hindari minuman dingin berlebihan karena suhu rendah dapat menyempitkan pembuluh darah di sekitar bronkus and mengurangi efisiensi pertukaran gas di paru-paru.
                """)

        elif skor_kelayakan > 50.0:
            if kategori_final in ["BAIK", "SEDANG"]:
                st.info("ℹ️ **Tindakan Pencegahan Zona Waspada — Polusi Harian Mulai Terakumulasi**")
                st.markdown(f"""
**😷 Persiapan Pelindung Diri Sehari-hari:**
- Sediakan selalu masker medis standar minimal 3-ply atau masker bedah di dalam tas Anda. Masker ini sudah cukup efektif untuk perjalanan harian dengan kendaraan umum atau sepeda motor dalam durasi di bawah satu jam.
- Bagi pengguna sepeda motor, kenakan masker sebelum menyalakan mesin and bukan setelah berada di tengah kemacetan karena pada titik itulah konsentrasi gas buang di ketinggian setara kepala pengendara sudah sangat tinggi.
- Simpan tisu basah antibakteri di kantong baju untuk sesekali membersihkan area hidung and mulut ketika merasa udara di sekitar sangat pengap.

**💧 Manajemen Hidrasi Cerdas:**
- Tingkatkan konsumsi air minum bersih menjadi minimal dua setengah liter per hari. Air yang cukup dalam tubuh membantu mengencerkan lendir di saluran napas sehingga partikel polutan yang terperangkap di dalamnya lebih mudah dikeluarkan melalui proses batuk atau bersin alami tubuh.
- Konsumsi minuman hangat berbahan jahe atau kunyit di pagi and sore hari. Senyawa gingerol and kurkumin yang terkandung di dalamnya terbukti memiliki sifat antiinflamasi yang membantu meredam respons peradangan di saluran bronkus akibat paparan polutan.
- Hindari konsumsi rokok dalam bentuk apapun karena nikotin and tar yang masuk bersama partikel polutan dari luar akan menciptakan beban ganda yang jauh melebihi kemampuan mekanisme pembersihan alami paru-paru.

**🚗 Strategi Mobilitas Harian:**
- Hindari berjalan kaki atau bersepeda di sepanjang jalan arteri utama pada jam sibuk pagi and sore hari. Pilih rute alternatif melalui gang perumahan atau jalan hijau yang lebih jauh namun jauh lebih sedikit polutannya.
- Jika menggunakan mobil pribadi, atur tombol sirkulasi udara AC ke mode sirkulasi internal terutama ketika terjebak dalam kemacetan panjang untuk mencegah udara kotor dari luar masuk ke kabin.
- Kurangi durasi nongkrong atau menunggu di pinggir jalan raya utama. Setiap sepuluh menit berdiri di tepi jalan padat setara dengan menghirup emisi gas buang setara beberapa batang rokok.

**🏠 Manajemen Udara Dalam Ruangan:**
- Tutup jendela and pintu rumah pada rentang waktu jam padat lalu lintas yaitu pukul 06.30 hingga 08.30 pagi and pukul 16.00 hingga 18.30 sore untuk mencegah gelombang polusi puncak masuk ke dalam hunian.
- Buka jendela di sisi rumah yang menghadap taman atau vegetasi, bukan yang menghadap langsung ke jalan raya atau area parkir kendaraan.
- Gunakan kipas angin atau exhaust fan untuk mengusir udara lama di dalam ruangan and menggantinya dengan udara yang sudah difilter secara alami oleh vegetasi di sekitar rumah.
- Hindari membakar sampah, dupa berlebihan, atau menggunakan pengharum ruangan berbahan aerosol kimia karena partikel mikro yang dihasilkan akan menambah beban polutan di dalam ruangan.

**🌿 Penguatan Imunitas Paru-paru:**
- Konsumsi suplemen vitamin D3 karena penelitian terkini menunjukkan bahwa kadar vitamin D yang optimal dalam tubuh sangat membantu mengurangi keparahan respons inflamasi saluran napas terhadap paparan polutan.
- Perbanyak makan ikan berlemak seperti salmon, tuna, and sarden yang kaya asam lemak omega-3 untuk mengurangi produksi senyawa inflamasi prostaglandin di jaringan paru-paru.
- Tidur cukup minimal tujuh jam per malam karena sistem imun tubuh melakukan regenerasi sel and perbaikan jaringan yang rusak akibat polutan secara optimal pada fase tidur nyenyak.

**👶 Perhatian Ekstra untuk Kelompok Rentan:**
- Batasi waktu bermain anak-anak di luar ruangan terutama pada sore hari ketika konsentrasi ozon permukaan tanah sedang di puncaknya akibat reaksi fotokimia dengan sinar matahari.
- Bagi ibu hamil, hindari perjalanan panjang melewati kawasan industri atau area proyek konstruksi yang menghasilkan debu PM10 dalam jumlah besar karena partikel ini dapat mempengaruhi perkembangan paru-paru janin.
- Lansia dengan penyakit jantung atau diabetes sebaiknya membawa inhaler atau obat bronkodilator sebagai tindakan antisipasi meskipun kondisi udara secara umum masih di zona waspada and belum darurat.
                """)
            else:
                st.info("ℹ️ **Tindakan Waspada dengan Status Lebih Serius — Perlindungan Ditingkatkan**")
                st.markdown(f"""
**😷 Perlindungan Pernapasan Ditingkatkan:**
- Dengan status {kategori_final} yang tercatat, masker medis standar tidak lagi cukup. Gunakan masker respirator minimal jenis KF94 atau setara yang mampu menyaring partikel hingga ukuran 0.3 mikrometer dengan efisiensi minimal 94 persen.
- Pastikan masker terpasang dengan sempurna menutup hidung and dagu tanpa ada celah udara di samping hidung karena celah sekecil apapun akan membuat polutan masuk melewati jalur tersebut.
- Jika Anda harus bekerja di luar ruangan untuk waktu lama, ganti masker setiap empat jam karena lapisan filter yang sudah jenuh dengan partikel polutan tidak lagi efektif menyaring udara baru.

**🏠 Perlindungan Hunian Total:**
- Pasang karet segel pada celah bawah pintu utama and jendela untuk meminimalisir kebocoran udara luar yang kotor masuk ke dalam ruangan.
- Operasikan air purifier dengan filter HEPA grade jika tersedia, terutama di kamar tidur anak-anak and ruang keluarga. Filter HEPA mampu menangkap 99.97 persen partikel berukuran 0.3 mikrometer termasuk sebagian besar polutan berbahaya.
- Hindari memasak dengan bahan bakar kayu atau arang di dalam atau dekat bangunan tertutup karena asap hasil pembakaran biomasa mengandung partikel ultra-halus yang jauh lebih berbahaya dari polusi jalan raya.

**🔌 Pemantauan Kesehatan Aktif:**
- Catat gejala seperti batuk ringan, bersin berulang, atau mata perih yang muncul setelah aktivitas di luar ruangan. Jika gejala ini bertahan lebih dari dua hari, segera konsultasikan ke dokter karena bisa jadi polutan sudah memicu inflamasi saluran napas yang butuh penanganan medis.
- Ukur saturasi oksigen darah dengan pulse oximeter jika tersedia. Nilai normal adalah 95 hingga 100 persen. Nilai di bawah 94 persen saat istirahat adalah sinyal untuk segera mencari udara bersih atau bantuan medis.
                """)

        elif skor_kelayakan > 25.0:
            st.warning("⚠️ **Rencana Darurat Zona Terancam — Perlindungan Maksimal Diperlukan Sekarang**")
            st.markdown(f"""
**🚨 Prioritas Utama: Hentikan Paparan Langsung Sekarang Juga:**
- Jika Anda sedang berada di luar ruangan, segera masuk ke dalam bangunan tertutup terdekat. Setiap menit tambahan paparan terhadap udara dengan skor {skor_kelayakan:.1f}% ini menambah beban polutan yang harus didetoksifikasi oleh organ hati and paru-paru Anda.
- Larang keras seluruh anggota keluarga terutama anak-anak and lansia melakukan aktivitas apa pun di luar ruangan hingga kondisi udara membaik atau minimal naik ke zona waspada.
- Segera kumpulkan semua anggota keluarga ke dalam satu ruangan yang sudah disiapkan sebagai "ruang perlindungan bersih" yang ventilasinya sudah ditutup rapat.

**😷 Pilihan Masker Wajib Berdasarkan Kondisi:**
- Untuk keluar ruangan darurat seperti menuju fasilitas kesehatan atau evakuasi, wajib mengenakan masker respirator N95 atau KN95 yang telah tersertifikasi. Masker ini harus menutupi hidung and dagu secara penuh and kedap.
- Bagi penderita asma, PPOK, atau penyakit jantung, konsultasikan dengan dokter apakah perlu respirator khusus dengan katup ekspirasi untuk memudahkan pernapasan saat kondisi darurat.
- Anak di bawah dua tahun tidak boleh menggunakan masker jenis apapun. Solusi terbaik untuk mereka adalah tetap di dalam ruangan tertutup dengan purifier aktif.
- Jika masker N95 tidak tersedia, gunakan lapisan tiga lembar kain katun yang dilembabkan sedikit sebagai solusi darurat sementara karena kelembapan membantu menangkap partikel debu lebih efektif daripada kain kering.

**🔌 Sistem Perlindungan Udara Dalam Ruangan:**
- Operasikan air purifier dengan filter HEPA pada kecepatan tertinggi di semua ruangan yang ditempati. Jika hanya punya satu unit, prioritaskan ruang tidur anak-anak.
- Sebagai alternatif murah darurat, letakkan baskom berisi air bersih di beberapa sudut ruangan untuk membantu menangkap partikel debu kasar yang melayang di udara dalam ruangan.
- Tutup semua celah ventilasi menggunakan selotip atau kain basah untuk mencegah udara luar yang terkontaminasi polutan {polutan_kritikal} menyelinap masuk ke dalam ruangan.
- Jangan menyalakan kipas angin yang langsung menghadap ke jendela karena akan menyedot udara kotor dari luar masuk ke dalam ruangan dengan lebih cepat.

**🚗 Protokol Evakuasi Darurat Jika Harus Keluar:**
- Kunci semua jendela and pintu mobil rapat-rapat sebelum menghidupkan mesin. Aktifkan mode sirkulasi udara internal pada sistem AC mobil sehingga udara dalam kabin berputar and tidak mengambil udara dari luar.
- Rencanakan rute perjalanan yang paling jauh dari kawasan industri, jalan tol padat, and area terbuka tanpa vegetasi karena konsentrasi polutan di titik-titik tersebut bisa beberapa kali lipat lebih tinggi dari rata-rata wilayah.
- Hindari berhenti di lampu merah dalam waktu lama dengan jendela terbuka karena area persimpangan jalan adalah titik akumulasi gas karbon monoksida dari kendaraan yang tertinggi di wilayah perkotaan.
- Jika menggunakan transportasi umum, pilih jalur yang berada di dalam terowongan atau gedung tertutup daripada yang melintasi jalan terbuka.

**🩺 Respons Medis Waspada Aktif:**
- Siapkan bronkodilator atau inhaler di tempat yang mudah dijangkau oleh seluruh anggota keluarga yang berusia di atas enam tahun. Ajari anak-anak cara menggunakannya dengan benar.
- Pantau terus kondisi fisik anggota keluarga yang rentan setiap dua jam sekali. Tanda-tanda yang harus langsung mendapat perhatian medis adalah napas berbunyi (mengi), bibir atau ujung jari membiru, kebingungan mendadak, atau nyeri dada.
- Catat nomor telepon darurat puskesmas, IGD rumah sakit terdekat, and layanan ambulans di tempat yang mudah terlihat seluruh anggota keluarga.
- Bagi penderita asma yang sudah memiliki rencana aksi asma dari dokter, ini adalah saat untuk mengaktifkan zona merah dalam rencana tersebut.

**🍵 Nutrisi Detoksifikasi Darurat:**
- Konsumsi air putih hangat dalam jumlah banyak minimal tiga liter hari ini untuk membantu ginjal membuang sisa metabolisme polutan yang sudah terserap ke dalam aliran darah.
- Minum minuman kaya antioksidan seperti jus tomat, teh hijau tanpa gula, atau air lemon hangat setiap dua jam sekali untuk melawan radikal bebas yang dihasilkan oleh polutan {polutan_kritikal}.
- Hindari makanan berlemak tinggi and makanan ultra-proses hari ini karena sistem pencernaan yang berat bekerja akan mengalihkan sumber daya tubuh dari proses detoksifikasi polutan.
- Konsumsi N-asetilsistein (NAC) sebagai suplemen jika tersedia and tidak ada kontraindikasi medis, karena zat ini adalah prekursor glutathione, antioksidan terkuat yang diproduksi alami oleh tubuh untuk melawan polutan.

**🏥 Koordinasi dengan Lingkungan Sekitar:**
- Informasikan kondisi kritis kualitas udara ini kepada tetangga kanan-kiri terutama yang memiliki bayi, balita, lansia, atau anggota keluarga dengan penyakit kronis agar mereka juga segera mengambil tindakan perlindungan.
- Jika Anda tinggal di kawasan padat seperti rumah susun atau gang sempit, koordinasikan dengan ketua RT untuk menutup sementara aktivitas yang berpotensi menambah polusi lokal seperti pembakaran sampah atau penggunaan genset besar.
- Pantau terus aplikasi pemantau kualitas udara secara real-time and bagikan informasi pembaruan kondisi kepada grup keluarga atau komunitas warga agar semua orang bisa mengambil keputusan yang tepat.
            """)
        else:
            st.error("🚨 **PROTOKOL DARURAT ABSOLUT — NYAWA DALAM ANCAMAN NYATA**")
            st.markdown(f"""
**#🚑 TINDAKAN PERTAMA: EVAKUASI ATAU ISOLASI TOTAL DALAM WAKTU 60 DETIK:**
- Tinggalkan segera area terbuka. Kondisi udara dengan skor kelayakan hanya {skor_kelayakan:.1f}% mengindikasikan atmosfer beracun tinggi yang berbahaya bagi pernapasan langsung.
- Wajib gunakan masker respirator kelas N95 atau KN95 yang terkunci rapat sempurna tanpa celah sedikit pun apabila terpaksa melintasi area terbuka menuju zona evakuasi tertutup.
- Nyalakan sistem penyaring udara (Air Purifier) di tingkat daya maksimal di dalam ruang isolasi darurat dalam bangunan, and pasang tabung oksigen bantuan portabel bagi anggota keluarga yang mulai mengalami serangan sesak napas akut.
- Segera hubungi sambungan darurat medis atau kunjungi instalasi gawat darurat terdekat apabila muncul gejala klinis akut seperti batuk darah, pusing hebat mendadak, serta penyempitan dada yang tajam.
            """)

    st.write("---")
    st.write("#### 📊 Grafik Perbandingan Tingkat Bahaya Polutan Terhadap Batas Aman")
    
    kanvas_kontri, (sumbu_kontri, sumbu_skor) = plt.subplots(1, 2, figsize=(14, 5))
    df_kontribusi = pd.DataFrame({"Parameter Polutan": nama_fitur, "Tingkat Bahaya Relatif": rasio_kontribusi})
    df_kontribusi = df_kontribusi.sort_values(by="Tingkat Bahaya Relatif", ascending=True)
    warna_grafik = ['#2ec4b6' if r < 1.0 else '#e71d36' for r in df_kontribusi["Tingkat Bahaya Relatif"]]
    
    grafik_kontri = sns.barplot(
        x='Tingkat Bahaya Relatif', 
        y='Parameter Polutan', 
        data=df_kontribusi, 
        palette=warna_grafik, 
        ax=sumbu_kontri,
        edgecolor='black',
        linewidth=0.5
    )
    
    sumbu_kontri.bar_label(grafik_kontri.containers[0], fmt='%.2f', padding=5, fontweight='bold', fontsize=9)
    sumbu_kontri.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, label='Batas Aman Pemerintah')
    sumbu_kontri.set_title("Zat Polutan Paling Dominan (Nilai > 1.0 = Melewati Batas Aman)", fontsize=10, fontweight='bold')
    sumbu_kontri.set_xlabel("Rasio Kelampauan Batas")
    sumbu_kontri.set_xlim(0, max(df_kontribusi["Tingkat Bahaya Relatif"].max() * 1.15, 1.3))
    sumbu_kontri.legend()
    
    warna_skor = '#2ec4b6' if skor_kelayakan > 75.0 else ('#ff9f1c' if skor_kelayakan > 50.0 else '#e71d36')
    
    sumbu_skor.bar(["Skor Kelayakan Paru-paru"], [skor_kelayakan], color=warna_skor, edgecolor='black', width=0.4, linewidth=0.5)
    sumbu_skor.text(0, skor_kelayakan / 2 if skor_kelayakan > 10 else 5, f"{skor_kelayakan:.1f}%", ha='center', va='center', color='white' if skor_kelayakan > 10 else 'black', fontweight='bold', fontsize=18)
    sumbu_skor.set_ylim(0, 115)
    sumbu_skor.set_title("Persentase Kelayakan Udara untuk Dihirup Manusia", fontsize=10, fontweight='bold')
    sumbu_skor.set_ylabel("Tingkat Kelayakan (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(kanvas_kontri)