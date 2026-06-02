import streamlit as st
import os
import joblib

direktori_halaman = os.path.dirname(os.path.abspath(__file__))
direktori_app_folder = os.path.dirname(direktori_halaman)
direktori_akar_proyek = os.path.dirname(direktori_app_folder)

rute_css = os.path.join(direktori_akar_proyek, "app", "assets", "style.css")
if os.path.exists(rute_css):
    with open(rute_css, "r") as berkas_css:
        st.markdown(f"<style>{berkas_css.read()}</style>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color: #ffffff; margin-bottom: 10px; font-size: 18px;'>🎯 Navigasi Sistem</h3>", unsafe_allow_html=True)

st.title("ℹ️ Informasi Metodologi Proyek Akhir")
st.write("---")
    
st.markdown("""
### 🔬 Penjelasan Metode Komputasi
Proyek Kelompok 14 mengadopsi standar metodologi **CRISP DM** (*Cross Industry Standard Process for Data Mining*). Kami mengawinkan dua kutub algoritma terkuat:
1. **Random Forest Classifier (Supervised):** Bertindak sebagai pasukan pohon keputusan utama yang bertugas memprediksi label formal kualitas udara dengan ketahanan akurasi tinggi terhadap data pencilan.
2. **K-Means Clustering (Unsupervised):** Bertindak sebagai pembagi batas wilayah spasial alami untuk menemukan klasterisasi unik sebaran polusi murni berdasarkan kedekatan jarak Euclidean matriks tanpa bias intervensi manusia.

### 📂 Sumber Penyelidikan Dataset
Data yang diolah merupakan catatan historis pemantauan parameter ambien kualitas udara resmi yang mencakup enam partikel gas berbahaya utama secara harian. Seluruh data latih klasifikasi telah diamankan kepadatannya menggunakan teknik oversampling terukur demi menghindari bias deteksi pada kelas minoritas darurat.
""")