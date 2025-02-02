# streamlit_app.py

import streamlit as st
import pickle

# Load the saved files (model and feature extraction)
model = pickle.load(open('logistic_regression.pkl', 'rb'))
feature_extraction = pickle.load(open('feature_extraction.pkl', 'rb'))

# Function to make predictions
def predict_mail(input_text):
    input_user_mail = [input_text]
    input_data_features = feature_extraction.transform(input_user_mail)
    prediction = model.predict(input_data_features)
    return prediction

# Streamlit UI components
st.title('Email Classifier')

# Input field for user email text
mail = st.text_area("Enter the email text:")

# Make prediction when button is clicked
if st.button("Classify"):
    if mail:
        predicted_mail = predict_mail(input_text=mail)
        
        # Display result
        if predicted_mail == 1:
            st.write("The email is classified as: **Spam**")
        else:
            st.write("The email is classified as: **Not Spam**")
    else:
        st.write("Please enter the email text to classify.")
