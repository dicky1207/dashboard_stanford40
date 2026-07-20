# app.py - Dark Mode + Aksen Emas | Layout Proporsional Dua Kolom
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Klasifikasi 40 Aktivitas Manusia",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS KUSTOM
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: #0e1117;
    }
    .main {
        padding: 0.5rem 1rem !important;
        max-width: 1400px !important;
        margin: 0 auto !important;
    }
    .header-gold {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #f0c27f;
        box-shadow: 0 4px 16px rgba(240, 194, 127, 0.1);
        margin-bottom: 0.6rem;
        animation: fadeInDown 0.6s ease;
    }
    .header-gold h1 {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0c27f;
        margin: 0;
    }
    .header-gold p {
        font-size: 0.8rem;
        color: #c9d1d9;
        opacity: 0.7;
        margin: 0;
    }
    .card-dark {
        background: rgba(22, 33, 62, 0.6);
        backdrop-filter: blur(6px);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        border: 1px solid rgba(240, 194, 127, 0.1);
        box-shadow: 0 2px 12px rgba(0,0,0,0.4);
        margin-bottom: 0.6rem;
        transition: transform 0.2s ease, box-shadow 0.3s ease;
        animation: fadeInUp 0.5s ease;
    }
    .card-dark:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(240, 194, 127, 0.06);
        border-color: rgba(240, 194, 127, 0.2);
    }
    .card-dark h3 {
        color: #f0c27f;
        font-weight: 600;
        font-size: 0.95rem;
        margin: 0 0 0.3rem 0;
        border-bottom: 1px solid rgba(240, 194, 127, 0.08);
        padding-bottom: 0.2rem;
    }
    .card-dark p, .card-dark li {
        color: #c9d1d9;
        font-size: 0.8rem;
        line-height: 1.3;
        margin: 0;
    }
    .stFileUploader > div {
        border: 2px dashed #f0c27f !important;
        border-radius: 10px !important;
        background: rgba(240, 194, 127, 0.03) !important;
        padding: 0.6rem !important;
        transition: all 0.3s ease;
    }
    .stFileUploader > div:hover {
        background: rgba(240, 194, 127, 0.08) !important;
        border-color: #ffd700 !important;
    }
    .stFileUploader label {
        color: #f0c27f !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    .metric-gold {
        background: rgba(22, 33, 62, 0.7);
        border-radius: 8px;
        padding: 0.3rem 0.2rem;
        text-align: center;
        border-left: 3px solid #f0c27f;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    .metric-gold .label {
        color: #8b949e;
        font-size: 0.6rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .metric-gold .value {
        color: #f0c27f;
        font-size: 1.1rem;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(240, 194, 127, 0.1);
    }
    .badge-gold {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.6rem;
        background: rgba(240, 194, 127, 0.1);
        color: #f0c27f;
        border: 1px solid rgba(240, 194, 127, 0.15);
        margin-top: 0.2rem;
    }
    .badge-high { background: #2d6a4f; color: #95d5b2; border-color: #52b788; }
    .badge-medium { background: #7f4f24; color: #f0c27f; border-color: #f0c27f; }
    .badge-low { background: #6b2d2d; color: #f28482; border-color: #e76f51; }
    .info-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem 1.5rem;
        background: rgba(22, 33, 62, 0.4);
        border-radius: 8px;
        padding: 0.4rem 1rem;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(240, 194, 127, 0.08);
        animation: fadeInUp 0.5s ease;
    }
    .info-item {
        color: #c9d1d9;
        font-size: 0.75rem;
    }
    .info-item strong {
        color: #f0c27f;
    }
    .streamlit-expanderHeader {
        color: #f0c27f !important;
        background: rgba(22, 33, 62, 0.4) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(240, 194, 127, 0.06);
        font-size: 0.8rem !important;
        padding: 0.2rem 0.6rem !important;
    }
    .streamlit-expanderContent {
        background: rgba(14, 17, 23, 0.8) !important;
        border-radius: 0 0 8px 8px !important;
        padding: 0.4rem 0.6rem !important;
    }
    .stDownloadButton button {
        background: linear-gradient(135deg, #f0c27f, #d4a574) !important;
        color: #0e1117 !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.2rem 1rem !important;
        font-weight: 700 !important;
        font-size: 0.7rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(240, 194, 127, 0.1);
    }
    .stDownloadButton button:hover {
        transform: scale(1.04);
        box-shadow: 0 4px 14px rgba(240, 194, 127, 0.25);
    }
    .footer-dark {
        text-align: center;
        color: #8b949e;
        font-size: 0.65rem;
        padding: 0.4rem 0;
        border-top: 1px solid rgba(240, 194, 127, 0.05);
        margin-top: 0.8rem;
    }
    .footer-dark span {
        color: #f0c27f;
    }
    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-12px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(12px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #f0c27f; border-radius: 6px; }
    ::-webkit-scrollbar-thumb:hover { background: #d4a574; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="header-gold">
    <h1>🏃 Pengenalan 40 Aktivitas Manusia</h1>
    <p>Stanford 40 Actions · ResNet50V2 + Transfer Learning</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model_and_classes():
    try:
        if not os.path.exists('har_40_model.keras'):
            st.error("❌ File model 'har_40_model.keras' tidak ditemukan!")
            return None, None
        if not os.path.exists('class_names.joblib'):
            st.error("❌ File 'class_names.joblib' tidak ditemukan!")
            return None, None
        with st.spinner("⏳ Memuat model..."):
            model = tf.keras.models.load_model('har_40_model.keras')
            class_names = joblib.load('class_names.joblib')
        return model, class_names
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None, None

model, CLASS_NAMES = load_model_and_classes()
if model is None or CLASS_NAMES is None:
    st.stop()

# ============================================================
# INFO BAR
# ============================================================
st.markdown(f"""
<div class="info-bar">
    <span class="info-item"><strong>Kelas:</strong> {len(CLASS_NAMES)} aktivitas</span>
    <span class="info-item"><strong>Arsitektur:</strong> ResNet50V2</span>
    <span class="info-item"><strong>Input:</strong> 224×224</span>
    <span class="info-item"><strong>Augmentasi:</strong> Cutout + Mixup</span>
    <span class="info-item"><strong>Akurasi:</strong> >70%</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# UPLOAD + PETUNJUK (2 KOLOM)
# ============================================================
col_upload, col_guide = st.columns([2, 1], gap="small")

with col_upload:
    with st.container():
        st.markdown('<div class="card-dark">', unsafe_allow_html=True)
        st.markdown("### 📤 Unggah Gambar")
        uploaded_file = st.file_uploader(
            "Seret atau klik untuk memilih gambar (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

with col_guide:
    with st.container():
        st.markdown('<div class="card-dark">', unsafe_allow_html=True)
        st.markdown("### ℹ️ Petunjuk")
        st.markdown("""
        <p style="margin:0; font-size:0.8rem;">
        1. Upload gambar aktivitas<br>
        2. Model akan prediksi aktivitas<br>
        3. Lihat hasil & probabilitas
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PREDIKSI (2 KOLOM: GAMBAR + HASIL)
# ============================================================
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        col_img, col_pred = st.columns([1, 1], gap="small")

        with col_img:
            with st.container():
                st.markdown('<div class="card-dark" style="text-align:center;padding:0.4rem;">', unsafe_allow_html=True)
                st.image(image, caption='📷 Gambar yang diunggah', use_container_width=False, width=300)
                st.markdown('</div>', unsafe_allow_html=True)

        with col_pred:
            with st.container():
                st.markdown('<div class="card-dark">', unsafe_allow_html=True)
                with st.spinner("🔮 Memprediksi..."):
                    time.sleep(0.2)
                    img_resized = image.resize((224, 224))
                    img_array = np.array(img_resized) / 255.0
                    img_input = np.expand_dims(img_array, axis=0)
                    pred_probs = model.predict(img_input, verbose=0)[0]
                    pred_class = np.argmax(pred_probs)
                    confidence = pred_probs[pred_class]

                st.markdown("### 📊 Hasil Prediksi")
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f"""
                    <div class="metric-gold">
                        <div class="label">🏷️ Aktivitas</div>
                        <div class="value">{CLASS_NAMES[pred_class]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-gold">
                        <div class="label">🎯 Keyakinan</div>
                        <div class="value">{confidence:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                if confidence >= 0.8:
                    badge = '<span class="badge-gold badge-high">✅ Keyakinan Tinggi</span>'
                elif confidence >= 0.5:
                    badge = '<span class="badge-gold badge-medium">⚠️ Keyakinan Sedang</span>'
                else:
                    badge = '<span class="badge-gold badge-low">❌ Keyakinan Rendah</span>'
                st.markdown(badge, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        # ============================================================
        # GRAFIK + TABEL DETAI L - DUA KOLOM SEJAJAR (proporsional)
        # ============================================================
        col_chart, col_table = st.columns([1, 1], gap="small")

        with col_chart:
            with st.container():
                st.markdown('<div class="card-dark">', unsafe_allow_html=True)
                st.markdown("### 📈 Distribusi Probabilitas (10 Kelas Teratas)")

                top_indices = np.argsort(pred_probs)[-10:][::-1]
                top_probs = pred_probs[top_indices]
                top_labels = [CLASS_NAMES[i] for i in top_indices]

                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('#16213e')
                bars = ax.barh(top_labels, top_probs, color='#f0c27f', edgecolor='#1a1a2e', linewidth=1)
                bars[0].set_color('#ffd700')
                ax.axvline(x=0.5, color='#d4a574', linestyle='--', linewidth=1.5, label='Threshold 50%')
                ax.set_xlabel('Probabilitas', color='#c9d1d9', fontsize=9)
                ax.set_title('10 Aktivitas Tertinggi', color='#f0c27f', fontsize=10, fontweight='600')
                ax.set_xlim(0, 1.05)
                ax.tick_params(colors='#c9d1d9', labelsize=8)
                ax.legend(loc='lower right', facecolor='#16213e', edgecolor='#f0c27f', labelcolor='#c9d1d9', fontsize=7)
                ax.grid(True, alpha=0.1, axis='x', color='#f0c27f')
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

        with col_table:
            with st.container():
                st.markdown('<div class="card-dark">', unsafe_allow_html=True)
                st.markdown("### 📋 Detail Probabilitas Semua Kelas")

                prob_df = pd.DataFrame({
                    'Aktivitas': CLASS_NAMES,
                    'Probabilitas': pred_probs
                }).sort_values('Probabilitas', ascending=False)

                st.dataframe(
                    prob_df.style.format({'Probabilitas': '{:.2%}'})
                    .background_gradient(cmap='viridis', subset=['Probabilitas']),
                    use_container_width=True,
                    height=350
                )

                csv = prob_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil (CSV)",
                    data=csv,
                    file_name='hasil_prediksi_aktivitas.csv',
                    mime='text/csv',
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

        if confidence < 0.5:
            st.info("💡 **Tips:** Coba upload gambar dengan pose yang lebih jelas atau latar belakang yang tidak terlalu ramai.")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {str(e)}")
        st.info("Coba upload gambar lain dengan format JPG atau PNG.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer-dark">
    🧠 Dibangun dengan <span>TensorFlow</span> &amp; <span>ResNet50V2</span> · Dataset Stanford 40 Actions
    <br>© 2026 Muhammad Dicky Imansyah · Magister Teknik Informatika
</div>
""", unsafe_allow_html=True)