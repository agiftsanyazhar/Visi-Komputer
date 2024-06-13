import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model("Image_Classification.keras")

data_cat = [
    "apple",
    "banana",
    "beetroot",
    "bell pepper",
    "cabbage",
    "capsicum",
    "carrot",
    "cauliflower",
    "chilli pepper",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "ginger",
    "grapes",
    "jalepeno",
    "kiwi",
    "lemon",
    "lettuce",
    "mango",
    "onion",
    "orange",
    "paprika",
    "pear",
    "peas",
    "pineapple",
    "pomegranate",
    "potato",
    "raddish",
    "soy beans",
    "spinach",
    "sweetcorn",
    "sweetpotato",
    "tomato",
    "turnip",
    "watermelon",
]

img_height = 180
img_width = 180

st.title("Fruit/Vegetable Image Classification")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image)
    img_batch = tf.expand_dims(img_arr, 0)

    # Predict the image
    predict = model.predict(img_batch)

    score = tf.nn.softmax(predict)

    st.write(
        "Fruit/Vegetable in image is **{}** with an accuracy of **{:.2f}%**".format(
            data_cat[np.argmax(score)].upper(), np.max(score) * 100
        )
    )
else:
    st.write("Please upload an image to classify.")
