
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Modelni yuklash
model = tf.keras.models.load_model("apple_leaf_model.h5")
class_names = ["Healthy", "Rust", "Scab"]

st.title("🍏 Apple Leaf Disease Classifier")
st.write("Upload a leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Rasmni ochish
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Model formatiga o'tkazish
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Bashorat
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.write("### Prediction:", class_names[class_index])
    st.write(f"### Confidence: {confidence:.2f}%")
