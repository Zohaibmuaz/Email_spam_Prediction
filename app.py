import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")  # Save your vectorizer too if needed, else use the same as in your script

# Title of the Streamlit app
st.title("SMS Spam Classifier")

# Description
st.write("This app classifies SMS messages as Ham (1) or Spam (0). Enter a message to classify.")

# Input text from the user
user_input = st.text_area("Enter your SMS message here:")

# When the button is clicked
if st.button("Classify"):
    if user_input:
        # Preprocess the input
        input_tfidf = vectorizer.transform([user_input])

        # Make prediction
        prediction = model.predict(input_tfidf)

        # Display the result
        result = "Spam" if prediction == 0 else "Ham"
        st.write(f"The message is: **{result}**")

    else:
        st.write("Please enter a message to classify.")
