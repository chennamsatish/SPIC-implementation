import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model

st.title("📰 News Category Prediction")

# Load saved objects
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")
model = load_model("news_model.h5")

# Text input
news_text = st.text_area("Enter News Text", height=200)

if st.button("Predict Category"):
    if news_text.strip() == "":
        st.warning("Please enter some news text")
    else:
        # Convert text to TF-IDF
        X = vectorizer.transform([news_text]).toarray()

        # Predict
        prediction = model.predict(X)
        predicted_index = np.argmax(prediction)

        # Decode label
        predicted_label = label_encoder.inverse_transform([predicted_index])

        st.success(f"Predicted Category: {predicted_label[0]}")
