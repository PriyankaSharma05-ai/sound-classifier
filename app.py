import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

model = tf.keras.models.load_model("Audio_Classification_Fixed/model/best_model.keras")

labels = ["Crow","Parrot","Sparrow"]

st.title("Bird Sound Classifier")

uploaded_file = st.file_uploader("Upload Bird Sound", type=["wav","mp3"])

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel = librosa.power_to_db(mel)

    mel = mel[:128,:130]
    mel = mel.reshape(1,128,130,1)

    prediction = model.predict(mel)
    predicted_label = labels[np.argmax(prediction)]

    st.success(f"Predicted Bird: {predicted_label}")