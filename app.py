import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Apple Doctor AI",
    page_icon="🍏",
    layout="centered"
)

# --- MODEL LOADING LOGIC ---
@st.cache_resource
def load_model_from_drive():
    model_path = 'apple_leaf_model.h5'
    
    # 1. Model fayli serverda bormi tekshiramiz
    if not os.path.exists(model_path):
        # Sizning Google Drive ID raqamingiz
        file_id = '1-xbx4i7Q5Qpnu2kwFtYjj_ce5nydO9ge'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        with st.spinner("Model Google Drive-dan yuklanmoqda (bir marta)..."):
            gdown.download(url, model_path, quiet=False)
    
    # 2. Modelni o'qish
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Modelni yuklashda xatolik yuz berdi: {e}")
        return None

# --- MAIN APP ---
def main():
    st.title("🍏 Apple Leaf Disease Classifier")
    st.markdown("---")
    
    # Modelni yuklash
    model = load_model_from_drive()
    class_names = ["Healthy (Sog'lom)", "Rust (Zang)", "Scab (Qo'tir)"]

    if model is None:
        st.warning("Model yuklanmadi. Iltimos, internetni yoki fayl ID sini tekshiring.")
        return

    # Fayl yuklash qismi
    uploaded_file = st.file_uploader("Barg rasmini yuklang (JPG, PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Ekranni ikkiga bo'lamiz: Chapda Rasm, O'ngda Natija
        col1, col2 = st.columns(2)

        with col1:
            # Rasmni ochish va RGB ga o'tkazish (muhim!)
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Yuklangan rasm", use_container_width=True)

        # Bashorat qilish
        with st.spinner('Tahlil qilinmoqda...'):
            # Model uchun tayyorlash
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Natijani olish
            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            result_text = class_names[class_index]

        # Natijani chiqarish
        with col2:
            st.subheader("Natija:")
            
            # Agar sog'lom bo'lsa yashil, kasal bo'lsa qizil rangda chiqaramiz
            if "Healthy" in result_text:
                st.success(f"✅ {result_text}")
            else:
                st.error(f"⚠️ {result_text}")
            
            st.metric("Ishonchlilik darajasi", f"{confidence:.2f}%")
            
            # Qo'shimcha maslahat
            if "Rust" in result_text:
                st.info("Maslahat: Zang kasalligi zamburug'li bo'lib, fungitsidlar bilan davolash tavsiya etiladi.")
            elif "Scab" in result_text:
                st.info("Maslahat: Qo'tir kasalligi namlik yuqori bo'lganda tarqaladi. Zararlangan barglarni olib tashlang.")
            elif "Healthy" in result_text:
                st.balloons()

if __name__ == "__main__":
    main()
