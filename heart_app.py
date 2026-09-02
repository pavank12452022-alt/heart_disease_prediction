import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("features.pkl")

st.title("Heart disease Prediction by pavan")
st.markdown("Provide the following details to check your heart disease risk:")

# Collect user input
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# When Predict is clicked
if st.button("Predict"):

    raw_input = {
        "age": age,
        "restingbp": resting_bp,
        "cholesterol": cholesterol,
        "fastingbs": fasting_bs,
        "maxhr": max_hr,
        "oldpeak": oldpeak,

        "sex_m": 1 if sex == "M" else 0,

        "chestpaintype_ata": 1 if chest_pain == "ATA" else 0,
        "chestpaintype_nap": 1 if chest_pain == "NAP" else 0,
        "chestpaintype_ta": 1 if chest_pain == "TA" else 0,

        "restingecg_normal": 1 if resting_ecg == "Normal" else 0,
        "restingecg_st": 1 if resting_ecg == "ST" else 0,

        "exerciseangina_y": 1 if exercise_angina == "Y" else 0,

        "st_slope_flat": 1 if st_slope == "Flat" else 0,
        "st_slope_up": 1 if st_slope == "Up" else 0
    }

    input_df = pd.DataFrame([raw_input])

    # Add any missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Exact same column order as training
    input_df = input_df[expected_columns]

    # Scale using training scaler
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")