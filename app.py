import pandas as pd
import streamlit as st
import joblib

# Load trained model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")          
label_encoders = joblib.load("label_encoders.pkl")  

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter your health and lifestyle details to predict the risk of Heart Disease.")

# ------------------ User Inputs ------------------
age = st.number_input("Age", 1, 120, 40)
gender = st.selectbox("Gender", ["Male", "Female"])
bp = st.number_input("Blood Pressure", 80, 200, 120)
family_hd = st.selectbox("Family Heart Disease", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0, 0.1)
hbp = st.selectbox("High Blood Pressure", ["No", "Yes"])
triglyceride = st.number_input("Triglyceride Level", 50, 500, 150)
fbs = st.number_input("Fasting Blood Sugar (mg/dL)", 50, 200, 100)
crp = st.number_input("CRP Level (mg/L)", 0.0, 20.0, 1.0, 0.1)
homocysteine = st.number_input("Homocysteine Level (µmol/L)", 0.0, 100.0, 10.0, 0.1)

chol_tc = st.number_input("Total Cholesterol", 100, 400, 200)
low_hdl = st.selectbox("Low HDL Cholesterol", ["No", "Yes"])
high_ldl = st.selectbox("High LDL Cholesterol", ["No", "Yes"])

sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
exercise = st.selectbox("Exercise Habits", ["Low", "Medium", "High"])
smoking = st.selectbox("Smoking", ["No", "Yes"])
stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])

sugar = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"])
alcohol = st.selectbox("Alcohol Consumption", ["Low", "Medium", "High"])

# ------------------ Preprocessing Functions ------------------
def cholesterol_ratio(tc, low_hdl, high_ldl):
    if tc < 200:
        tc_bucket = 0
    elif tc < 240:
        tc_bucket = 1
    else:
        tc_bucket = 2

    if low_hdl == "Yes" and high_ldl == "Yes":
        return 3 + tc_bucket
    elif low_hdl == "Yes" and high_ldl == "No":
        return 2 + tc_bucket
    elif low_hdl == "No" and high_ldl == "Yes":
        return 1 + tc_bucket
    else:
        return tc_bucket

def sleep_score(hours):
    if 7 <= hours <= 9: return 2
    elif 5 <= hours < 7 or 9 < hours <= 11: return 1
    else: return 0

def exercise_score(level): return {"Low":0,"Medium":1,"High":2}[level]
def smoking_score(level): return {"No":2,"Yes":0}[level]
def stress_score(level): return {"Low":2,"Medium":1,"High":0}[level]

# ------------------ Transform Inputs ------------------
# Lifestyle Index
lifestyle_index = sleep_score(sleep_hours) + exercise_score(exercise) + smoking_score(smoking) + stress_score(stress)

# Label encode categorical columns
gender_enc = label_encoders["Gender"].transform([gender])[0]
family_hd_enc = label_encoders["Family Heart Disease"].transform([family_hd])[0]
diabetes_enc = label_encoders["Diabetes"].transform([diabetes])[0]
hbp_enc = label_encoders["High Blood Pressure"].transform([hbp])[0]


# One-hot encode sugar/alcohol
sugar_high = 1 if sugar=="High" else 0
sugar_medium = 1 if sugar=="Medium" else 0
sugar_low = 1 if sugar=="Low" else 0

alcohol_high = 1 if alcohol=="High" else 0
alcohol_medium = 1 if alcohol=="Medium" else 0
alcohol_low = 1 if alcohol=="Low" else 0

# Cholesterol ratio
chol_ratio = cholesterol_ratio(chol_tc, low_hdl, high_ldl)

# ------------------ Construct input DataFrame ------------------
input_df = pd.DataFrame({
    "Age":[age],
    "Gender":[gender_enc],
    "Blood Pressure":[bp],
    "Family Heart Disease":[family_hd_enc],
    "Diabetes":[diabetes_enc],
    "BMI":[bmi],
    "High Blood Pressure":[hbp_enc],
    "Triglyceride Level":[triglyceride],
    "Fasting Blood Sugar":[fbs],
    "CRP Level":[crp],
    "Homocysteine Level":[homocysteine],
    "Cholesterol Ratio":[chol_ratio],
    "Lifestyle_Index":[lifestyle_index],
    "Sugar Consumption_High":[sugar_high],
    "Sugar Consumption_Low":[sugar_low],
    "Sugar Consumption_Medium":[sugar_medium],
    "Alcohol Consumption_High":[alcohol_high],
    "Alcohol Consumption_Low":[alcohol_low],
    "Alcohol Consumption_Medium":[alcohol_medium]
})

# Normalize numeric columns using saved scaler
numeric_cols = ["Age","Blood Pressure","BMI","Triglyceride Level","Fasting Blood Sugar", "CRP Level","Homocysteine Level"]
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# ------------------ Prediction ------------------
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ The model predicts: **Heart Disease Detected**")
    else:
        st.success("✅ The model predicts: **No Heart Disease**")
