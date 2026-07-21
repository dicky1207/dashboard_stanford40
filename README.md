# **🏃 Pengenalan 40 Aktivitas Manusia (Stanford 40 Actions)**

Proyek ini adalah *pipeline Machine Learning* lengkap (Mulai dari Persiapan Data, Pelatihan Model, hingga Antarmuka Web) untuk mengklasifikasikan 40 jenis aktivitas manusia berdasarkan gambar.

Aplikasi ini menggunakan model **Deep Learning** berbasis arsitektur **ResNet50V2** dengan teknik *Transfer Learning* tingkat lanjut, yang kemudian disajikan melalui antarmuka (Dashboard) **Streamlit** berdesain elegan (Dark Mode \+ Aksen Emas).

## **✨ Fitur Utama**

* **🛠️ Pipeline Pelatihan Lengkap:** Skrip terpisah dan terstruktur untuk *Data Preparation* dan *Model Training*.  
* **🧠 Arsitektur Model Canggih:** Menggunakan **ResNet50V2** dengan dua tahap pelatihan (*Frozen Base* & *Fine-Tuning*).  
* **🎨 Teknik Augmentasi Modern:** Mengimplementasikan **Cutout** (Random Erasing) dan **Mixup** secara *custom* melalui Data Generator untuk mencegah *overfitting* dan meningkatkan akurasi.  
* **⚖️ Penanganan Data Imbalance:** Menggunakan *Balanced Class Weights* secara otomatis.  
* **💻 Dashboard Interaktif & Elegan:** Antarmuka Streamlit kustom dengan tema *Dark Mode* dan aksen emas. Menampilkan metrik probabilitas, grafik distribusi (Top 10), dan opsi unduh hasil prediksi dalam format CSV.

## **🛠️ Teknologi yang Digunakan**

* **Bahasa Pemrograman:** Python 3.10.12  
* **Deep Learning Framework:** TensorFlow & Keras  
* **Web Dashboard:** Streamlit  
* **Data Science & Machine Learning:** Scikit-learn, Numpy, Pandas, Joblib  
* **Visualisasi:** Matplotlib, Seaborn

## **📂 Struktur Direktori Proyek**

Agar skrip dapat berjalan dengan baik, pastikan struktur direktori Anda seperti berikut setelah mengekstrak dataset:

dashboard\_stanford40/  
│  
├── app.py                   \# Skrip Dashboard Streamlit  
├── prepare\_data.py          \# Skrip pembagian dataset (80:20)  
├── train.py                 \# Skrip pelatihan model (ResNet50V2)  
├── requirements.txt         \# Daftar dependensi library  
├── class\_names.joblib       \# (Digenerate otomatis) Daftar kelas  
├── har\_40\_model.keras       \# (Digenerate otomatis) Model hasil training  
│  
├── stanford40/              \# ⬅️ ANDA HARUS MELETAKKAN DATASET ASLI DI SINI  
│   ├── appalauding\_001.jpg  
│   ├── brushing\_teeth\_001.jpg  
│   └── ...  
│  
└── data/                    \# (Digenerate otomatis) Struktur folder untuk training  
    ├── train/  
    └── test/

## **🚀 Panduan Instalasi & Penggunaan**

Ikuti langkah-langkah di bawah ini untuk melatih model dan menjalankan dashboard dari awal:

### **1\. Persiapan Awal**

Kloning repositori ini dan buat *virtual environment*:

git clone https://github.com/dicky1207/dashboard_stanford40.git  
cd dashboard\_stanford40

\# Buat virtual environment & aktifkan (opsional namun disarankan)  
python \-m venv venv  
\# Windows:  
venv\\Scripts\\activate  
\# Mac/Linux:  
source venv/bin/activate

\# Instal semua library yang dibutuhkan  
pip install \-r requirements.txt

Unduh dataset **Stanford 40 Actions** dari [situs resminya](http://vision.stanford.edu/Datasets/40actions.html). Ekstrak gambar-gambar tersebut ke dalam sebuah folder bernama stanford40 di dalam direktori proyek ini.

### **2\. Persiapan Data (Data Split)**

Jalankan skrip persiapan data. Skrip ini akan membagi data menjadi 80% data latih dan 20% data uji, serta menata ulang ke dalam struktur folder yang siap dibaca oleh Keras.

python prepare\_data.py

*Proses ini akan menghasilkan folder data/ dan file class\_names.joblib.*

### **3\. Pelatihan Model (Training)**

Mulai latih model ResNet50V2 Anda. Skrip ini menjalankan 2 tahap pelatihan (*Base model* dan *Fine-tuning*) dengan augmentasi *Cutout* dan *Mixup*.

python train.py

*Tunggu hingga proses selesai. Ini mungkin memakan waktu lama tergantung pada GPU Anda. Hasilnya berupa model har\_40\_model.keras serta grafik evaluasi training\_curves.png dan confusion\_matrix.png.*

### **4\. Menjalankan Dashboard**

Setelah model berhasil dilatih dan disimpan, jalankan antarmuka web Streamlit:

streamlit run app.py

Aplikasi akan otomatis terbuka di browser Anda (biasanya pada http://localhost:8501). Silakan unggah gambar baru untuk melihat model memprediksi aktivitas\!

## **📊 Metrik Evaluasi Proyek**

Skrip pelatihan akan otomatis memplot **Akurasi & Loss per Epoch** dan menghasilkan **Confusion Matrix** dari data uji (*Test set*). Silakan cek gambar yang dihasilkan di folder proyek setelah pelatihan selesai untuk menganalisa performa (target akurasi rata-rata \>70%).

## **👨‍💻 Kredit**

Dikembangkan oleh [**Muhammad Dicky Imansyah**](https://instagram.com/_monarch12/) (Copyright © 2026 \- Magister Teknik Informatika \- UPI YPTK).
