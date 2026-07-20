import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# ============================================================
# 1. LOAD MODEL DAN NAMA KELAS (DENGAN ERROR HANDLING)
# ============================================================
st.set_page_config(page_title="Klasifikasi 40 Aktivitas Manusia", layout="centered")

@st.cache_resource
def load_model_and_classes():
    """Memuat model dan nama kelas dengan error handling."""
    try:
        # Cek keberadaan file
        if not os.path.exists('har_40_model.keras'):
            st.error("❌ File model 'har_40_model.keras' tidak ditemukan!")
            st.info("Pastikan file model ada di direktori yang sama dengan app.py")
            return None, None
        
        if not os.path.exists('class_names.joblib'):
            st.error("❌ File 'class_names.joblib' tidak ditemukan!")
            st.info("Pastikan file class_names.joblib ada di direktori yang sama")
            return None, None
        
        # Load model dan nama kelas
        model = tf.keras.models.load_model('har_40_model.keras')
        class_names = joblib.load('class_names.joblib')
        
        return model, class_names
    
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None, None

# Muat model (dengan cache)
model, CLASS_NAMES = load_model_and_classes()

if model is None or CLASS_NAMES is None:
    st.stop()  # Hentikan eksekusi jika model gagal dimuat

# ============================================================
# 2. KONFIGURASI APLIKASI
# ============================================================
st.title("🏃 Pengenalan 40 Aktivitas Manusia (Stanford 40)")
st.markdown("""
**Unggah gambar** seseorang yang sedang melakukan aktivitas, dan model akan memprediksi aktivitas tersebut.
""")

# Informasi model
with st.expander("ℹ️ Informasi Model"):
    st.write(f"- **Jumlah kelas:** {len(CLASS_NAMES)} aktivitas")
    st.write(f"- **Arsitektur:** MobileNetV2 (transfer learning)")
    st.write(f"- **Ukuran input:** 224×224 piksel")
    st.write("- **Akurasi uji:** 70% (pada dataset Stanford 40 Actions)")

# ============================================================
# 3. UPLOAD GAMBAR
# ============================================================
uploaded_file = st.file_uploader(
    "📤 Pilih gambar (JPG/PNG)...", 
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# 4. PROSES PREDIKSI
# ============================================================
if uploaded_file is not None:
    try:
        # --- 4a. Baca dan tampilkan gambar ---
        image = Image.open(uploaded_file)
        
        # Validasi: cek apakah gambar berwarna (RGB)
        if image.mode != 'RGB':
            st.warning(f"⚠️ Gambar dalam mode {image.mode}, akan dikonversi ke RGB.")
            image = image.convert('RGB')
        
        # Tampilkan gambar
        st.image(image, caption='📷 Gambar yang diunggah', width=250)
        
        # --- 4b. Preprocessing ---
        with st.spinner('🔮 Memprediksi...'):
            # Resize sesuai input model
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_input = np.expand_dims(img_array, axis=0)
            
            # --- 4c. Prediksi ---
            pred_probs = model.predict(img_input, verbose=0)[0]
            pred_class = np.argmax(pred_probs)
            confidence = pred_probs[pred_class]
        
        # --- 4d. Validasi prediksi ---
        if confidence < 0.2:
            st.warning("⚠️ Tingkat keyakinan rendah (<20%). Model mungkin tidak yakin dengan prediksi ini.")
        
        # ============================================================
        # 5. VISUALISASI HASIL
        # ============================================================
        
        # --- 5a. Hasil utama ---
        st.subheader("📊 Hasil Prediksi")
        col1, col2 = st.columns(2)
        col1.metric("🏷️ Aktivitas", CLASS_NAMES[pred_class])
        col2.metric("🎯 Tingkat Keyakinan", f"{confidence:.2%}")
        
        # --- 5b. Grafik probabilitas 10 kelas teratas ---
        st.subheader("📈 Distribusi Probabilitas (10 Kelas Teratas)")
        
        top_indices = np.argsort(pred_probs)[-10:][::-1]
        top_probs = pred_probs[top_indices]
        top_labels = [CLASS_NAMES[i] for i in top_indices]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(top_labels, top_probs, color='skyblue')
        
        # Highlight prediksi utama
        bars[0].set_color('forestgreen')
        
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='Threshold 50%')
        ax.set_xlabel('Probabilitas', fontsize=11)
        ax.set_title('10 Aktivitas dengan Probabilitas Tertinggi', fontsize=12)
        ax.set_xlim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.2, axis='x')
        
        st.pyplot(fig)
        
        # --- 5c. Tabel detail semua kelas ---
        with st.expander("📋 Lihat Detail Probabilitas Semua Kelas"):
            prob_df = pd.DataFrame({
                'Aktivitas': CLASS_NAMES,
                'Probabilitas': pred_probs
            }).sort_values('Probabilitas', ascending=False)
            
            # Tambahkan styling
            st.dataframe(
                prob_df.style.format({'Probabilitas': '{:.2%}'})
                .background_gradient(cmap='Blues', subset=['Probabilitas'])
            )
            
            # Download hasil prediksi
            csv = prob_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Hasil Prediksi (CSV)",
                data=csv,
                file_name='hasil_prediksi_aktivitas.csv',
                mime='text/csv'
            )
        
        # --- 5d. Rekomendasi jika keyakinan rendah ---
        if confidence < 0.5:
            st.info("💡 **Tips:** Coba upload gambar dengan pose yang lebih jelas atau latar belakang yang tidak terlalu ramai.")
    
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses gambar: {str(e)}")
        st.info("Coba upload gambar lain dengan format JPG atau PNG.")

# ============================================================
# 6. FOOTER
# ============================================================
st.markdown("---")
st.caption("🧠 Dibangun dengan TensorFlow + MobileNetV2 | Dataset Stanford 40 Actions")