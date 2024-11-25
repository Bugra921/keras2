import uvicorn
import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras
from pathlib import Path
import cv2
import streamlit as st

MODEL = keras.models.load_model("my_model.keras")
CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Resmi uygun boyuta dönüştürün
    img = img.astype('float32') / 255.0  # Normalizasyon yapın
    img = np.expand_dims(img, 0)  # Batch boyutuna göre yeniden şekillendirin
    return img

camera_input = st.camera_input('Kameradan resim çek')
gallery_input = st.file_uploader('VEYA Fasulye Fotoğrafı Ekleyin', accept_multiple_files=False)

if camera_input is not None:
    img_bytes = camera_input.getvalue()
    img = Image.open(BytesIO(img_bytes))
    img_cv2 = np.array(img)
    
    img_batch = preprocess_image(img_cv2)
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    st.title({
        "class": predicted_class,
        "confidence": float(confidence)
    })

elif gallery_input is not None:
    img_bytes = gallery_input.getvalue()
    img = Image.open(BytesIO(img_bytes))
    img_cv2 = np.array(img)
    
    img_batch = preprocess_image(img_cv2)
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    st.title({
        "class": predicted_class,
        "confidence": float(confidence)
    })

else:
    st.write("Lütfen bir resim yükleyin veya kamera kullanarak bir resim çekin.")
