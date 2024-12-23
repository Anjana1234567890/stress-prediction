# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:51:34 2024

@author: Anjana
"""

import streamlit as st
import joblib
import numpy as np

# Load the saved model and vectorizer
model = joblib.load("C:/Users/Anjana/Desktop/stress_detection_new/stress_detection_model.pkl")  # Update the path
vectorizer = joblib.load("C:/Users/Anjana/Desktop/stress_detection_new/tfidf_vectorizer.pkl")  # Update the path

# Create the Streamlit app interface
st.title('Stress Detection using ML')
st.write('Enter some text, and the model will predict if the person is stressed or not.')

# Text input field
user_input = st.text_area("Enter text for stress detection:")

# Prediction button
if st.button('Predict Stress Level'):
    if user_input:
        # Transform the user input using the loaded vectorizer
        input_tfidf = vectorizer.transform([user_input])

        # Make the prediction
        prediction = model.predict(input_tfidf)

        # Show the prediction result
        if prediction[0] == 1:
            st.success("The person is stressed!")
        else:
            st.success("The person is not stressed.")

    else:
        st.warning("Please enter some text to get the prediction.")
