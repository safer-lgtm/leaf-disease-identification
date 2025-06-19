import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json

# Model laden
model = tf.keras.models.load_model('outputs/best_model.h5')
#model.summary()

# Labels laden
with open("data/class_indices.json", "r") as f:
    class_indices = json.load(f)
    index_to_class = {int(v): k for k, v in class_indices.items()}

# Vorhersage-Funktion
def predict_image(model, img_data):
    img = load_img(img_data, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    probs = model.predict(img)
    top_prob = float(np.max(probs))
    top_pred = index_to_class[int(np.argmax(probs))]
    return top_prob, top_pred

# Streamlit UI
st.set_page_config(page_title="Plant Disease Classifier", layout="centered")
st.title("🌿Pflanzenkrankheiten - VGG16")

#st.image("data/logo.jpeg", width=200)

uploaded_file = st.file_uploader("Wähle ein Blattbild (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Hochgeladenes Bild", width=300)

    prob, pred = predict_image(model, uploaded_file)

    st.markdown(f"### Vorhersage: **{pred}**")
    st.progress(int(prob * 100))
    st.markdown(f"**Modellkonfidenz:** {round(prob * 100, 2)}%")
else:
    st.info("Bitte ein Bild hochladen, um eine Vorhersage zu erhalten.")