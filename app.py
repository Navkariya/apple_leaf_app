
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import gdown

@st.cache_resource
def load_model_from_drive():
    # Model fayli borligini tekshirish, bo'lmasa yuklab olish
    if not os.path.exists('apple_leaf_model.h5'):
        # Bu yerga o'z ID raqamingizni yozing
        file_id = '1-xbx4i7Q5Qpnu2kwFtYjj_ce5nydO9ge'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, 'apple_leaf_model.h5', quiet=False)
    
    model = tf.keras.models.load_model('apple_leaf_model.h5')
    return model

# Modelni yuklash
model = load_model_from_drive()
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
