"""
Dashboard Streamlit - Klasifikasi Citra Emosi Wajah dengan CNN (MobileNetV2)
Jalankan:  streamlit run app.py
Prasyarat: notebook sudah dijalankan sampai selesai sehingga folder `output/` terisi.
"""
import os
import json

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------
st.set_page_config(page_title="Klasifikasi Emosi Wajah", page_icon="🙂", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
MODEL_PATH = os.path.join(OUT_DIR, "final_emotion_cnn_model.keras")
CLASS_PATH = os.path.join(OUT_DIR, "class_names.json")
SUMMARY_PATH = os.path.join(OUT_DIR, "dataset_summary.json")
METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")
HISTORY_PATH = os.path.join(OUT_DIR, "history.json")
CM_PATH = os.path.join(OUT_DIR, "confusion_matrix.npy")

CONF_THRESHOLD = 50  # di bawah ini tampilkan peringatan

EMOJI = {
    "angry": "😠", "anger": "😠", "contempt": "😒", "disgust": "🤢", "fear": "😨",
    "happy": "😄", "happiness": "😄", "neutral": "😐", "sad": "😢", "sadness": "😢",
    "surprise": "😮", "surprised": "😮",
}

st.markdown(
    """
    <style>
    .hero {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.8rem 1.5rem;
           border-radius: 18px; text-align: center; margin-bottom: 1.2rem;
           box-shadow: 0 6px 20px rgba(0,0,0,.15);}
    .hero h1 {margin: 0; color: white; font-size: 2.1rem;}
    .hero p {margin: .4rem 0 0; color: #ece8ff;}
    .result-card {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: #06331f;
           padding: 1.2rem; border-radius: 16px; text-align: center;
           box-shadow: 0 6px 18px rgba(17,153,142,.35);}
    .result-card .emoji {font-size: 3.2rem;}
    .result-card .name {font-size: 1.9rem; font-weight: 700; text-transform: capitalize;}
    .result-card .conf {font-size: 1.1rem;}
    </style>
    <div class="hero">
      <h1>Klasifikasi Citra Emosi Wajah dengan CNN</h1>
      <p>Convolutional Neural Network (MobileNetV2) &middot; Dashboard Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# LOAD ASSET (di-cache)
# ------------------------------------------------------------
def load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


@st.cache_resource(show_spinner="Memuat model...")
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH) as f:
        names = json.load(f)
    return model, names


if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASS_PATH)):
    st.error("Model belum ditemukan. Jalankan dulu notebook sampai selesai agar folder `output/` terisi "
             "(harus ada `final_emotion_cnn_model.keras` dan `class_names.json`).")
    st.stop()

model, class_names = load_model_and_classes()
summary = load_json(SUMMARY_PATH, {})
metrics = load_json(METRICS_PATH, {})
hist_blob = load_json(HISTORY_PATH, {})
IMG_SIZE = tuple(model.input_shape[1:3])


def predict(img: Image.Image):
    """Resize, normalisasi 0-1 (Rescaling ke [-1,1] ada di dalam model), lalu prediksi dengan flip-TTA."""
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32") / 255.0
    batch = np.stack([arr, arr[:, ::-1, :]], axis=0)
    probs = model.predict(batch, verbose=0).mean(axis=0)
    return probs


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Tentang Model")
    st.write("**Arsitektur:** MobileNetV2 (transfer learning)")
    st.write(f"**Input:** {IMG_SIZE[0]}×{IMG_SIZE[1]} piksel")
    st.write(f"**Jumlah kelas:** {len(class_names)}")
    st.markdown("**Kelas yang dikenali:**")
    for c in class_names:
        st.write(f"{EMOJI.get(c, '•')} {c.capitalize()}")
    st.divider()
    st.caption("Tips: gunakan foto wajah yang jelas, menghadap kamera, dengan pencahayaan cukup.")

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab_ring, tab_eval, tab_pred = st.tabs(["📊 Ringkasan & Dataset", "📈 Evaluasi Model", "🔮 Prediksi Gambar Baru"])

# ---------- TAB 1: Ringkasan dataset ----------
with tab_ring:
    st.subheader("Deskripsi Proyek")
    st.write(
        "Proyek ini membangun model **Convolutional Neural Network (CNN)** untuk mengklasifikasikan emosi wajah "
        "dari citra. Dataset berasal dari Kaggle (*Facial Emotion Recognition Dataset*). Model memakai transfer "
        "learning MobileNetV2 dengan dua fase training (classifier head lalu fine-tuning)."
    )
    st.caption("Sumber data: https://www.kaggle.com/datasets/fahadullaha/facial-emotion-recognition-dataset")

    if summary:
        total = summary.get("total_images", 0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah kelas", summary.get("num_classes", len(class_names)))
        c2.metric("Total gambar", f"{total:,}")
        c3.metric("Data latih", f"{summary.get('n_train', 0):,}")
        c4.metric("Data uji", f"{summary.get('n_test', 0):,}")
        if total:
            st.write(
                f"**Proporsi data:** latih {summary['n_train'] / total * 100:.1f}% · "
                f"validasi {summary['n_val'] / total * 100:.1f}% · uji {summary['n_test'] / total * 100:.1f}% "
                "(validasi diambil dari bagian latih untuk early stopping)."
            )

        st.subheader("Distribusi Data per Kelas")
        counts = pd.Series(summary.get("class_counts", {}), name="Jumlah gambar")
        st.bar_chart(counts)
        st.caption(f"Rasio kelas terbanyak : tersedikit = {summary.get('imbalance_ratio', 0):.2f}x. "
                   "Ketidakseimbangan dikompensasi dengan class weight saat training.")

        with st.expander("Parameter pelatihan"):
            st.json({
                "Ukuran gambar": summary.get("img_size"),
                "Batch size": summary.get("batch_size"),
                "Optimizer": summary.get("optimizer"),
                "Learning rate (fase 1 / fase 2)": [summary.get("lr_head"), summary.get("lr_fine")],
                "Epoch dijalankan (fase 1 / fase 2)": [summary.get("epochs_phase1_run"), summary.get("epochs_phase2_run")],
                "Layer di-fine-tune": summary.get("fine_tune_layers"),
                "Total parameter": summary.get("total_params"),
                "Lama training (menit)": summary.get("train_minutes"),
            })
    else:
        st.info("`dataset_summary.json` belum ada di folder `output/`.")

    p = os.path.join(OUT_DIR, "1b_sample_per_class.png")
    if os.path.exists(p):
        st.subheader("Contoh Gambar per Kelas")
        st.image(p, use_container_width=True)

# ---------- TAB 2: Evaluasi ----------
with tab_eval:
    if metrics:
        st.subheader("Hasil Evaluasi pada Data Uji")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        m2.metric("Precision (macro)", f"{metrics['precision_macro']:.4f}")
        m3.metric("Recall (macro)", f"{metrics['recall_macro']:.4f}")
        m4.metric("F1-score (macro)", f"{metrics['f1_macro']:.4f}")
        st.caption(
            f"Weighted -> Precision {metrics['precision_weighted']:.4f} · Recall {metrics['recall_weighted']:.4f} "
            f"· F1 {metrics['f1_weighted']:.4f} · jumlah data uji: {metrics['n_test']}"
        )
        rows = [{"Kelas": c, "Precision": v["precision"], "Recall": v["recall"],
                 "F1-score": v["f1-score"], "Support": int(v["support"])}
                for c, v in metrics["per_class"].items()]
        st.dataframe(pd.DataFrame(rows).style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}),
                     use_container_width=True, hide_index=True)
    else:
        st.info("`metrics.json` belum ada di folder `output/`.")

    if hist_blob.get("history"):
        st.subheader("Grafik Akurasi dan Loss Selama Training")
        h = hist_blob["history"]
        cA, cL = st.columns(2)
        cA.line_chart(pd.DataFrame({"Train": h["accuracy"], "Validation": h["val_accuracy"]}))
        cA.caption("Accuracy per epoch")
        cL.line_chart(pd.DataFrame({"Train": h["loss"], "Validation": h["val_loss"]}))
        cL.caption("Loss per epoch")
        st.caption(f"Fine-tuning dimulai setelah epoch ke-{hist_blob.get('phase_split', '?')}.")

    if os.path.exists(CM_PATH):
        st.subheader("Confusion Matrix")
        cm = np.load(CM_PATH)
        cm_df = pd.DataFrame(cm, index=[f"Aktual: {c}" for c in class_names],
                             columns=[f"Pred: {c}" for c in class_names])
        st.dataframe(cm_df.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)
    for fname, title in [("3_confusion_matrix.png", "Confusion Matrix (gambar)"),
                         ("3b_f1_per_class.png", "F1-score per Kelas"),
                         ("5_gradcam.png", "Feature Importance (Grad-CAM)"),
                         ("4_sample_predictions.png", "Contoh Prediksi pada Data Uji")]:
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, use_container_width=True)

# ---------- TAB 3: Prediksi ----------
with tab_pred:
    st.subheader("Uji Prediksi Gambar Baru")
    mode = st.radio("Sumber gambar", ["Upload", "Kamera"], horizontal=True)
    if mode == "Upload":
        file = st.file_uploader("Pilih foto wajah", type=["jpg", "jpeg", "png", "webp"])
    else:
        file = st.camera_input("Ambil foto wajah")

    if file is not None:
        image = Image.open(file)
        probs = predict(image)
        order = np.argsort(probs)[::-1]
        top = int(order[0])
        name = class_names[top]
        conf = float(probs[top]) * 100

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.image(image, caption="Gambar input", use_container_width=True)
        with col2:
            st.markdown(
                f"""
                <div class="result-card">
                  <div class="emoji">{EMOJI.get(name, '🙂')}</div>
                  <div class="name">{name}</div>
                  <div class="conf">Keyakinan: {conf:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if conf < CONF_THRESHOLD:
                st.warning("Keyakinan rendah. Coba foto wajah yang lebih jelas atau menghadap kamera.")
            else:
                st.success("Prediksi cukup meyakinkan.")

        st.subheader("Probabilitas semua kelas")
        for i in order:
            pr = float(probs[i])
            st.write(f"{EMOJI.get(class_names[i], '•')} **{class_names[i].capitalize()}** — {pr * 100:.1f}%")
            st.progress(min(max(pr, 0.0), 1.0))
    else:
        st.info("Belum ada gambar. Upload foto wajah atau gunakan kamera untuk mulai.")

st.divider()
st.caption("Dibuat dengan TensorFlow & Streamlit")
