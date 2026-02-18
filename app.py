import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --- SAHIFA SOZLAMALARI ---
st.set_page_config(
    page_title="Apple Doctor AI",
    page_icon="🍏",
    layout="centered"
)

# --- MODELNI YUKLASH FUNKSIYASI ---
@st.cache_resource
def load_model_smart():
    model_path = 'apple_leaf_model.h5'
    # Sizning Google Drive ID raqamingiz (Code ichiga yozib qo'ydim)
    file_id = '1gpD-kjaczzbDBiqehShs2Yv-Vm8Fo_EV'
    url = f'https://drive.google.com/uc?id={file_id}'

    # 1. Funksiya: Faylni yuklab olish
    def download_file():
        if os.path.exists(model_path):
            os.remove(model_path) # Eski fayl bo'lsa o'chiramiz
        with st.spinner("Model Google Drive-dan yuklanmoqda (taxminan 128 MB)..."):
            gdown.download(url, model_path, quiet=False)

    # 2. Asosiy mantiq: Fayl yo'q bo'lsa, yuklaymiz
    if not os.path.exists(model_path):
        download_file()

    # 3. Modelni o'qishga urinib ko'ramiz
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        # AGAR XATO BERESA (Demak fayl chala yuklangan)
        st.warning(f"Fayl buzilgan ko'rinadi ({e}). Qayta yuklanmoqda...")
        
        # Buzuq faylni o'chirib, qayta yuklaymiz
        download_file()
        
        # Ikkinchi urinish
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e2:
            st.error(f"Kechirasiz, modelni yuklab bo'lmadi. Xatolik: {e2}")
            return None

# --- ASOSIY DASTUR ---
def main():
    st.title("🍏 Apple Leaf Disease Classifier")
    st.write("Olma bargi rasmini yuklang va sun'iy intellekt tashxis qo'yadi.")
    st.markdown("---")

    # Modelni chaqiramiz
    model = load_model_smart()
    class_names = ["Healthy (Sog'lom)", "Rust (Zang)", "Scab (Qo'tir)"]

    if model is None:
        st.stop() # Model bo'lmasa dastur to'xtaydi

    # Rasm yuklash tugmasi
    uploaded_file = st.file_uploader("Rasmni tanlang...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Ekranni ikkiga bo'lamiz
        col1, col2 = st.columns(2)

        with col1:
            # Rasmni ochish va RGB ga o'tkazish
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Yuklangan rasm", use_container_width=True)

        # Bashorat qilish jarayoni
        with col2:
            with st.spinner('Tahlil qilinmoqda...'):
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                
                result_text = class_names[class_index]

            st.subheader("Natija:")
            
            # Rangli xabarlar
            if "Healthy" in result_text:
                st.success(f"✅ {result_text}")
                st.balloons()
            else:
                st.error(f"⚠️ {result_text}")
            
            st.metric("Ishonchlilik", f"{confidence:.2f}%")

            # Tavsiyalar
            if "Rust" in result_text:
                st.info("💡 **Tavsiya:** Zang kasalligiga qarshi fungitsidlar ishlating va zararlangan barglarni yo'q qiling.")
            elif "Scab" in result_text:
                st.info("💡 **Tavsiya:** Qo'tir kasalligi namlikda ko'payadi. Bog'ni shamollatishni yaxshilang.")

if __name__ == "__main__":
    main()
