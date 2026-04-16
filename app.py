import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration for a centered layout and custom title
st.set_page_config(page_title="Diabetes Risk Assessment", layout="wide")

# Load the model
model = joblib.load('model.pkl')

# Custom CSS to center elements and improve styling
st.markdown("""
    <style>
    .main {
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007BFF;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Centered Title and Description
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🩺 Health Diagnosis Portal")
    st.write("Enter the patient's clinical details below to assess diabetes risk.")
    st.divider()

# Organizing inputs into columns for a better look
with st.container():
    c1, c2 = st.columns(2)
    
    with c1:
        preg = st.slider("Number of Pregnancies", 0, 20, 1)
        gluc = st.number_input("Glucose Level (mg/dL)", 0, 250, 100)
        bp = st.number_input("Blood Pressure (mm Hg)", 0, 150, 70)
        skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)

    with c2:
        ins = st.number_input("Insulin Level (mu U/ml)", 0, 900, 80)
        bmi = st.number_input("Body Mass Index (BMI)", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.slider("Age (Years)", 1, 120, 30)

# Centering the Prediction Logic
st.divider()
left, mid, right = st.columns([1, 1, 1])

with mid:
    if st.button("Generate Diagnostic Report"):
        # Prepare feature vector
        features = np.array([[preg, gluc, bp, skin, ins, bmi, dpf, age]])
        
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)

        # Mapping numerical output to categorical values
        result_map = {0: "Healthy (Negative)", 1: "At Risk (Positive)"}
        final_result = result_map[prediction[0]]
        
        # Display Results
        st.subheader("Result:")
        if prediction[0] == 1:
            st.error(f"Prediction: {final_result}")
        else:
            st.success(f"Prediction: {final_result}")

        st.write(f"**Confidence:** {np.max(prediction_proba)*100:.2f}%")

        # Visualization of probabilities
        st.write("Confidence Breakdown:")
        prob_df = pd.DataFrame(prediction_proba, columns=["Healthy", "At Risk"])
        st.bar_chart(prob_df.T)
