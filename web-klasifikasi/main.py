# === Web App Sederhana untuk Klasifikasi Gambar Ikan ===
# Menggunakan Streamlit

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# === Load Model ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_ikan.h5')  # Ganti dengan path model Anda
    return model

model = load_model()

# === Set Judul Web ===
st.title("Klasifikasi Jenis Ikan dengan CNN (ResNet50)")
st.write("Upload gambar ikan, dan model akan memprediksi jenisnya.")

# === Upload Gambar ===
uploaded_file = st.file_uploader("Pilih gambar ikan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # Prediksi
    prediction = model.predict(img_preprocessed)
    class_index = np.argmax(prediction)

    # Mapping label (ganti sesuai label asli Anda)
    class_labels = ['Ikan A', 'Ikan B', 'Ikan C', 'Ikan D', 'Ikan E']
    pred_label = class_labels[class_index]

    st.success(f"Prediksi: {pred_label} ({prediction[0][class_index]*100:.2f}%)")
