# 1. Import library, Load model, dan nama kelas
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import pandas as pd


model = tf.keras.models.load_model('har_40_model.keras')
CLASS_NAMES = joblib.load('class_names.joblib')

# 2. Konfigurasi Halaman dan Upload Gambar
st.set_page_config(page_title="Klasifikasi 40 Aktivitas Manusia", layout="centered")
st.title("Pengenalan 40 Aktivitas Manusia (Dataset Stanford 40)")
st.write("Unggah gambar seseorang yang sedang melakukan aktivitas, dan model akan memprediksi aktivitas tersebut.")

uploaded_file = st.file_uploader("Pilih gambar (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca dan tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', width=250)
    
# 3. Preprocessing dan Prediksi
    # Preprocessing
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    
    # Prediksi
    pred_probs = model.predict(img_input)[0]
    pred_class = np.argmax(pred_probs)
    confidence = pred_probs[pred_class]

# 4. Visualisasi Hasil Prediksi    
    # Tampilkan hasil utama
    st.subheader("📊 Hasil Prediksi")
    col1, col2 = st.columns(2)
    col1.metric("Aktivitas", CLASS_NAMES[pred_class])
    col2.metric("Tingkat Keyakinan", f"{confidence:.2%}")
    
    # Grafik batang probabilitas 10 kelas teratas
    st.subheader("📈 Distribusi Probabilitas (10 Kelas Teratas)")
    top_indices = np.argsort(pred_probs)[-10:][::-1]
    top_probs = pred_probs[top_indices]
    top_labels = [CLASS_NAMES[i] for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top_labels, top_probs, color='skyblue')
    ax.axvline(x=0.5, color='red', linestyle='--', label='Threshold 50%')
    ax.set_xlabel('Probabilitas')
    ax.set_title('10 Aktivitas dengan Probabilitas Tertinggi')
    ax.legend()
    st.pyplot(fig)
    
    # Tabel detail semua kelas
    with st.expander("Lihat Detail Probabilitas Semua Kelas"):
        prob_df = pd.DataFrame({
            'Aktivitas': CLASS_NAMES,
            'Probabilitas': pred_probs
        }).sort_values('Probabilitas', ascending=False)
        st.dataframe(prob_df.style.format({'Probabilitas': '{:.2%}'}))