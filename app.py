import streamlit as st
import numpy as np
import joblib

# Load mock model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🫀 CardioGuard: Heart Disease Risk Predictor")
st.markdown("Predict your heart disease risk using AI. Just input your vitals below.")

with st.form("input_form"):
    age = st.number_input("Age", min_value=10, max_value=100, value=45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["Yes", "No"])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    sex_val = 1 if sex == "Male" else 0
    cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
    fbs_val = 1 if fbs == "Yes" else 0

    input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, thalach]])
    risk = model.predict_proba(input_data)[0][1] * 100

    st.subheader("🩺 Result")
    if risk < 40:
        st.success(f"Your risk score is {risk:.2f}%. Low risk. Stay healthy! 💚")
    elif risk < 70:
        st.warning(f"Your risk score is {risk:.2f}%. Moderate risk. Consider checkup. 🟡")
    else:
        st.error(f"Your risk score is {risk:.2f}%. High risk! Consult a doctor ASAP. 🔴")

    st.markdown("""### 💡 Prevention Tips
- Exercise regularly (30 min/day)  
- Eat heart-healthy foods  
- Reduce salt & sugar  
- Quit smoking  
- Manage stress  
- Regular health checkups""")
