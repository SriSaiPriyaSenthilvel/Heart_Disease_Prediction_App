# streamlit_app.py

import streamlit as st
import joblib
import numpy as np
import os

# Check if the model file exists
model_path = "diabetes_model.pkl"

# Check the current working directory and print it for debugging
st.write("Current working directory:", os.getcwd())

# Load the trained model if the file exists
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"Model file '{model_path}' not found. Please ensure it is in the correct directory.")
    st.stop()

# Function to make predictions
def make_prediction(input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI components
st.title('Diabetes Prediction App')

# Input fields
st.write("Please enter the following details:")

# Create input fields for user data
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
insulin = st.number_input("Insulin Level", min_value=0, max_value=500, value=85)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)

# Create the input_data array
input_data = np.array([[age, bmi, glucose, insulin, blood_pressure]])

# Make prediction when button is clicked
if st.button("Predict"):
    prediction = make_prediction(input_data)
    
    # Display result
    if prediction == 1:
        st.write("The model predicts: **Diabetic**")
    else:
        st.write("The model predicts: **Not Diabetic**")
