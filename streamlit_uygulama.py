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


MODEL = keras.models.load_model("mymodel.h5")
CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

camera_input = st.camera_input('Kameradan resim çek')
gallery_input = st.file_uploader(' VEYA Fasulye Fotoğrafı Ekleyin', accept_multiple_files=False)    
    
if camera_input is not None:
    img_bytes = camera_input.getvalue()
    img = Image.open(BytesIO(img_bytes))
    img_cv2 = np.array(img)
    
    img_cv2 = cv2.resize(img_cv2, (256,256))
    img_batch = np.expand_dims(img_cv2, 0)
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    st.title({
        "class": predicted_class,
        "confidence": float(confidence)
    })
elif gallery_input is not None and camera_input is None:
    img_bytes = gallery_input.getvalue()
    img = Image.open(BytesIO(img_bytes))
    img_cv2 = np.array(img)
    
    img_cv2 = cv2.resize(img_cv2, (256,256))
    img_batch = np.expand_dims(img_cv2, 0)
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    st.title({
        "class": predicted_class,
        "confidence": float(confidence)
    })
