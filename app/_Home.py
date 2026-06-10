import streamlit as st
import os

st.set_page_config(
    page_title="Sistem Cerdas Kualitas Udara Kelompok 14",
    layout="wide",
    initial_sidebar_state="expanded"
)

direktori_file = os.path.dirname(os.path.abspath(__file__))
rute_css = os.path.join(direktori_file, "assets", "style.css")

st.sidebar.markdown("<h3 style='color: #ffffff; margin-bottom: 10px; font-size: 18px;'>🎯 Navigasi Sistem</h3>", unsafe_allow_html=True)

if os.path.exists(rute_css):
    with open(rute_css, "r") as berkas_css:
        st.markdown(f"<style>{berkas_css.read()}</style>", unsafe_allow_html=True)

st.title("🎛️ Sistem Cerdas Analisis Kualitas Udara")
st.subheader("Pendekatan Integrasi Klasifikasi Supervised dan Klasterisasi Unsupervised")
st.write("---")
st.markdown("""
### 📌 Deskripsi Singkat Proyek
Sistem informasi cerdas ini dibangun untuk mendeteksi secara presisi tingkat risiko bahaya polusi udara berdasarkan sebaran zat polutan beracun di lingkungan. Melalui implementasi metodologi CRISP DM yang disiplin, proyek ini mampu menyajikan dua sudut pandang analisis sekaligus yaitu prediksi kategori formal pemerintah dan pemetaan karakteristik wilayah secara alami guna memberikan alarm peringatan dini bagi masyarakat luas.

### 👥 Identitas Anggota Kelompok 14
* **Farrel Farica Firjaturazza** (NIM 24051214034) — S1 Sistem Informasi Universitas Negeri Surabaya
* **Tata Ivanka** (NIM 24051214001) — S1 Sistem Informasi Universitas Negeri Surabaya
""")
