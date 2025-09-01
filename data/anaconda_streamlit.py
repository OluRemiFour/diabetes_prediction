import numpy as np
import streamlit as st
import pickle

import os
import pickle

# Absolute path (so it's consistent no matter where you run Streamlit)
MODEL_PATH = r"C:\Users\Dev\Desktop\py\data\trained_diabetes_model.sav"

# 1️⃣ Check if file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# 2️⃣ Check if file is readable
if not os.access(MODEL_PATH, os.R_OK):
    raise PermissionError(f"Model file is not readable: {MODEL_PATH}")

# 3️⃣ Load the model safely
with open(MODEL_PATH, "rb") as f:
    try:
        loaded_model = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle file. Error: {e}")

# Extract model + scaler
if not isinstance(loaded_model, dict):
    raise TypeError("Expected a dictionary with 'model' and 'scaler' keys, but got something else.")

if "model" not in loaded_model or "scaler" not in loaded_model:
    raise KeyError("Pickle file does not contain expected 'model' and 'scaler' keys.")

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