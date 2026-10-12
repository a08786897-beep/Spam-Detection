import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Models aur vectorizer load karein
model = joblib.load("logistic.pkl")
scaler = joblib.load("scaler.pkl")
vectorizer =joblib.load("columns.pkl")  # columns.pkl ki jagah vectorizer


# Predict message function define karein
def predict_message(message):
  # Text ko vectorize karein
  vec_text = vectorizer.transform([message])
  # Scale karein
  scaled_text = scaler.transform(vec_text)
  # Prediction karein
  prediction = model.predict(scaled_text)

  # Result display karne ke liye
  if prediction[0] == 0:
    st.error("🚨 This is a Spam message!")
  else:
    st.success("✅ This is not a Spam message.")


st.title("Spam Detection App")
st.markdown(
    "This app predicts whether a given message is spam or not using a"
    " pre-trained logistic regression model."
)

st.text_area("Enter your message here:", height=150, key="message_input")


# Button with prediction call
if st.button("Predict"):
  message = st.session_state.message_input
  if message.strip() != "":
    predict_message(message)
  else:
    st.warning("Please enter a message!")