import streamlit as st
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import gdown
import PIL
from PIL import Image
import numpy as np


st.set_page_config(
    page_title="Flower Image Classification App",
    page_icon="Flower",
    layout="centered"
)
st.title("Image Classification Using Convolutional Neural Netwrok")
st.subheader("-For Oxford Flowers")
st.write("This application uses Convolutional Neural Network(CNN)**"
         "to classifify real-world flower images")

# #Load Flower Label name
# ds_info=tfds.load(
#     "Oxford_flowers102",
#     with_info=True,
#     as_supervised=True
# )
# class_name=ds_info.featuers['label'].names
# class_names=["pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower", "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet William", "carnation", "garden phlox", "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula", "sunflower", "pelargonium", "bishop of Llandaff", "gaura", "geranium", "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus", "toad lily", "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper", "blackberry lily"]
class_names=[
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily"
]

#Download Model
MODEL_PATH="model.keras"

# if not os.path.exists(MODEL_PATH):
#     with st.spinner("Downloading trained model...."):
#         file_id="1_zETafXo_CaaAhft8ADgfbg5EojNcasa",
#         url=f"https://drive.google.com/file/d/1_zETafXo_CaaAhft8ADgfbg5EojNcasa/view?usp=sharing{file_id}",
#         gdown.download(url,MODEL_PATH,quiet=False)

#Load Model
MODEL_PATH="oxford_flower_model.keras"
file_id="1_zETafXo_CaaAhft8ADgfbg5EojNcasa"
download_url=f"https://drive.google.com/uc?id={file_id}"

#@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with st.spinner("Downloading Trained Model...."):
            gdown.download(download_url,MODEL_PATH,quiet=False)
            
    return tf.keras.models.load_model(MODEL_PATH)

model=load_model()

#Process Image
def preprocess_image(img):
    img=img.convert("RGB")
    img=img.resize((180,180))
    img_arr=np.array(img)/255.0
    img_arr=np.expand_dims(img_arr,axis=0)
    return img_arr


#Upload Image on Screen
uploaded_file=st.file_uploader(
    "Upload Flower Image",
    type=['jpg','jpeg','png']
)

#Prediction Logic
if uploaded_file is not None:
    img=Image.open(uploaded_file)

    st.image(img,
             caption="Image Uploaded",
             use_container_width=True)
    
    with st.spinner("Classifyng Image..."):
        processed_image=preprocess_image(img)
        #prediction=model.predict(processed_image)
        preds=model.predict(processed_image)
        probabilities=tf.nn.softmax(preds[0]).numpy()
        pred_index=np.argmax(probabilities)
        predicted_flower=class_names[pred_index]
        confidence=probabilities[pred_index]*100
        # predicted_class=np.argmax(prediction)
        # confidence=np.max(tf.nn.softmax(prediction))*100

    st.success(f"Predicted Flower : **{class_names[predicted_flower]}**")
    st.info(f"Confidence : **{confidence: .2f}%**")

#Footer
st.markdown("------")
st.caption(
    "CNN Model trained on Oxford 102 Flower Dataset"
    " - Developed by Mansi Rajapkar, Nitish Singh, & Nitish Jha"
)