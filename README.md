# 🎛️ Sistem Cerdas Analisis Kualitas Udara (Kelompok 14)

Sistem informasi cerdas berbasis web yang mengintegrasikan pendekatan **Klasifikasi Terawasi (Supervised Learning)** dan **Klasterisasi Tak Terawasi (Unsupervised Learning)** untuk mendeteksi tingkat risiko bahaya polusi udara secara dinamis. Proyek ini disusun menggunakan metodologi **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) sebagai pemenuhan Tugas Akhir mata kuliah Data Mining, Program Studi S1 Sistem Informasi, Universitas Negeri Surabaya.

Tautan Aplikasi Live: [https://kualitas-udara-kelompok-14kdd1.streamlit.app/](https://kualitas-udara-kelompok-14kdd1.streamlit.app/)
---

## 📌 Deskripsi Singkat Proyek
Masalah penurunan kualitas udara di kawasan urban akibat penumpukan partikel debu halus dan emisi gas beracun merupakan ancaman serius bagi kesehatan masyarakat. Sistem cerdas ini dibangun untuk menyajikan dua sudut pandang analisis sekaligus dari data polutan harian:

1. **Prediksi Kategori Resmi Pemerintah:** Menggunakan algoritma **Random Forest Classifier** untuk menetapkan status formal hukum kualitas udara (BAIK, SEDANG, TIDAK SEHAT, SANGAT TIDAK SEHAT) berdasarkan acuan regulasi Permen LHK No. 14 Tahun 2020.
2. **Pemetaan Karakteristik Wilayah Alami:** Menggunakan algoritma **K-Means Clustering** untuk memetakan klasterisasi profil kimia atmosfer di lapangan berdasarkan kedekatan jarak Euclidean matriks tanpa bias intervensi batas kaku hukum manusia.

Sistem juga dilengkapi dengan kalkulasi otomatis untuk menghitung parameter polutan paling dominan, tingkat kelayakan paru-paru, serta menyajikan **Panduan Aksi Nyata Lindungi Keluarga** yang bersifat adaptif dan taktis medis berdasarkan skor yang dihasilkan.

---

## 👥 Identitas Anggota Kelompok 14
* **Farrel Farica Firjaturazza** (NIM 24051214034) — S1 Sistem Informasi Universitas Negeri Surabaya
* **Tata Ivanka** (NIM 24051214001) — S1 Sistem Informasi Universitas Negeri Surabaya

---

## 🏗️ Struktur Modul Aplikasi
Sistem ini dibangun menerapkan prinsip modularitas tinggi agar kode program bersih, terstruktur, dan mudah dirawat:

* **🏠 Home:** Gerbang utama antarmuka yang menyajikan ringkasan proyek.
* **📊 Dataset Overview:** Modul tinjauan umum, validitas data, dan audit higienitas dataset.
* **🔮 Prediction Analysis:** Dasbor eksekusi model klasifikasi dan analisis klasterisasi dengan rekomendasi medis taktis.
* **📈 Visualization:** Galeri infografis tren temporal dan sebaran polutan untuk analisis Data Science mendalam.
* **ℹ️ About:** Profil transparan tim pengembang, visi misi, serta dokumentasi akuntabilitas riset.

---

## 📐 Arsitektur Repositori Proyek
```text
Tugas Akhir/
├── app/
│   ├── _Home.py                  # Gerbang Utama Antarmuka
│   ├── assets/
│   │   └── style.css             # Injeksi CSS Kustom
│   └── pages/
│       ├── 1_Dataset_Overview.py
│       ├── 2_Prediction_Analysis.py
│       ├── 3_Visualization.py
│       └── 4_About.py
├── dataset/
│   └── data_kualitas_udara.csv   # Berkas Data Utama
├── model/                        # Objek Model ML & Skalar (PKL)
│   ├── cluster_label_map.pkl
│   ├── model_classification_random_forest.pkl
│   ├── model_clustering_kmeans.pkl
│   ├── scaler_classification.pkl
│   └── scaler_clustering.pkl
├── notebook/
│   └── DTMG_KELOMPOK_14.ipynb    # Dokumentasi Riset Jupyter Notebook
└── requirements.txt              # Daftar Pustaka Server Cloud
