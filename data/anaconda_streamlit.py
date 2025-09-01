import numpy as np
import streamlit as st
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "trained_diabetes_model.sav")  # ✅ fixed

with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)

model = loaded_model["model"]
scaler = loaded_model["scaler"]

def diabetes_prediction(input_data):

    # Convert to numpy and reshape (1 row, 8 features)
    input_array = np.asarray(input_data).reshape(1, -1)

    # Apply SAME scaler
    std_input = scaler.transform(input_array)

    # Predict
    prediction = model.predict(std_input)

    print("Prediction:", prediction)  # [0] or [1]
    if prediction[0] == 0:
        return("The person is NOT diabetic")
    else:
        return("The person is diabetic")

def main():
    
    # giving title
    st.title('Diabetes Prediction Web App')
    
    # getting the input data from the user 
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPredigreeFunction = st.text_input('Diabetes Predigree Function value')
    Age = st.text_input('Age of the Person')
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for prediction 
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPredigreeFunction, Age]) 
    
    st.success(diagnosis)