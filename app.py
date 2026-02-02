import os
import gdown

MODEL_PATH = "flower_model.keras"

if not os.path.exists(MODEL_PATH):
    url = "https://1drv.ms/u/c/376e03e0a6ed20bb/IQBMC0AH6t0lQ6qSKdgEbgX8AeAHMSgIyQqGywKzD6U3BRo?e=6xJXtG"
    gdown.download(url, MODEL_PATH, quiet=False)

import tensorflow as tf
import streamlit as st 
from PIL import Image
import numpy as np 

# Load Trained Model 

def load_model():
    return tf.keras.models.load_model("oxford_flower_model.keras")

model=load_model()

# Class Name
class_names=["f class{i}" for i in range(102)]

#Image Size 
IMG_SIZE = 180

#Image Processing Function 
def preprocess_img(img):
    img=img.convert("RGB")
    img=img.resize((IMG_SIZE,IMG_SIZE))
    img_array=np.array(img)/255.0
    img_array=np.expand_dims(img_array)
    return img_array

#Build UI 
st.title("OXFORD FLOWER IMAGE CLASSIFICATION")
st.write("Upload a flower image to classify it ")

#File uploader 
uploaded_file=st.file_uploader("Choose an Image",
                               type=["jpg","jpeg","png"])

#Prediction Logic
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image, caption="Uploaded image",width=300)
    img_array=preprocess_img(image)

#Model Prediction
predictions=model.predict(img_array)

predicted_class=np.argmax(predictions)
confidence=np.max(tf.nn.softmax(predictions))*100

#Display Result
st.success(f"Prediction : {class_names[predicted_class]}")
st.info(f"Confidence : {confidence: .2f}%")

#Rum The App 
