import streamlit as st
import numpy as np
import gdown
import os
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

st.set_page_config(page_title="Animal Classifier", page_icon="🌟", layout="centered")
# Load model
file_id = "1_Vs62gZbs8LGKHNEEHLqnoyxG2heTQ9N"
model_path = "MCAR.keras"
# Download model if not present
if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    st.sidebar.write("Downloading model... Please wait.")
    gdown.download(url, model_path, quiet=False)
# Load the trained model
model = keras.models.load_model(model_path)

# Define class names
class_names = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']

st.title("Animal Classifier 🐾")
st.write("Upload an image of an animal and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded Image.', use_container_width=True)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}%**")

st.markdown(
    """
    <style>
    .bottom-credit {
        position: fixed;
        bottom: 10px;
        left: 0;
        width: 100%;
        text-align: right;
        font-size: 14px;
        color: white;
        padding: 5px;
    }
    </style>
    <div class="bottom-credit">
        Done by Gokul 🚀
    </div>
    """, unsafe_allow_html=True
)
