import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("diabetes_model.pkl")

# Streamlit UI
st.title("🩺 Diabetes Prediction App")

# User Input
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, step=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
age = st.number_input("Age", min_value=1, max_value=120, step=1)

# Predict button
if st.button("Predict"):
    input_features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                insulin, bmi, dpf, age]])
    prediction = model.predict(input_features)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes!")
    else:
        st.success("✅ Low Risk of Diabetes")
