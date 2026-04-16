import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model
model = joblib.load('model.pkl')

st.title("Diabetes Prediction App")
st.write("""
This app predicts the likelihood of diabetes based on clinical parameters.
""")

st.sidebar.header("User Input Parameters")

def user_input_features():
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=250, value=100)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

if st.button("Predict"):
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    st.subheader('Prediction')
    result = 'Positive' if prediction[0] == 1 else 'Negative'
    st.write(f"The model predicts a **{result}** result.")

    st.subheader('Prediction Probability')
    st.write(pd.DataFrame(prediction_proba, columns=['Negative', 'Positive']))
