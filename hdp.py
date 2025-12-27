import streamlit as st
import numpy as np
import pickle

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Heart Disease Risk Analyzer",
    page_icon="🫀",
    layout="wide"
)

# ================= LOAD MODEL =================
with open("hdp.pkl", "rb") as f:
    model = pickle.load(f)

# ================= SIDEBAR =================
st.sidebar.title("🫀 HeartCare AI")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 Project Insights", "🩺 Risk Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Developer:** Kamalesh")
st.sidebar.caption("ML-powered healthcare system")

# ================= GLOBAL STYLES =================
st.markdown("""
<style>
.card {
    background: rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.big {
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div style="background:linear-gradient(90deg,#e53935,#e35d5b);
            padding:25px;border-radius:18px">
<h1 style="color:white;text-align:center;">🫀 Heart Disease Risk Analyzer</h1>
<h4 style="color:white;text-align:center;">
AI-driven cardiovascular risk assessment
</h4>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# 🏠 OVERVIEW
# =====================================================
if page == "🏠 Overview":
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧠 Why Heart Disease Prediction?")
        st.write("""
        Heart disease remains the leading cause of death globally.
        Early detection enables preventive care and lifestyle intervention.
        """)

        st.subheader("🚨 Key Risk Factors")
        st.markdown("""
        - Hypertension & Cholesterol  
        - Smoking & Alcohol  
        - Diabetes & Obesity  
        - Stress & Sleep deprivation  
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Global Heart Deaths / Year", "17.9M")
        st.metric("Early Prevention Success", "80%")
        st.metric("Model Accuracy", "85%")
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# 📊 PROJECT INSIGHTS
# =====================================================
elif page == "📊 Project Insights":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📌 Project Summary")
    st.write("""
    This system predicts heart disease risk using supervised Machine Learning
    trained on clinical health data.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🧾 Input Features")
        st.markdown("""
        Age, Gender, BMI  
        Blood Pressure  
        Cholesterol  
        Smoking, Diabetes  
        Stress & Sleep  
        """)

    with col2:
        st.warning("⚙️ Tech Stack")
        st.markdown("""
        Python  
        NumPy & Pandas  
        Scikit-Learn  
        Streamlit  
        """)

    with col3:
        st.success("📈 ML Details")
        st.markdown("""
        Algorithm: Random Forest  
        Accuracy: 85%  
        Evaluation: ROC-AUC  
        """)

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# 🩺 RISK PREDICTION
# =====================================================
elif page == "🩺 Risk Prediction":

    st.subheader("🧾 Patient Health Information")

    with st.expander("ℹ️ How inputs affect prediction"):
        st.write("""
        • High BP & Cholesterol increase risk  
        • Smoking & Diabetes strongly affect heart health  
        • Proper sleep & low stress reduce risk  
        """)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 0, 120, 35)
        gender = st.selectbox("Gender", ["Female", "Male"])
        bmi = st.number_input("BMI", 0.0, 60.0, 24.0)

    with col2:
        bp = st.number_input("Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol Level", 100, 400, 200)
        sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)

    with col3:
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        stress = st.slider("Stress Level", 0, 10, 4)

    # ================= ENCODING =================
    gender = 1 if gender == "Male" else 0
    smoking = 1 if smoking == "Yes" else 0
    diabetes = 1 if diabetes == "Yes" else 0

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🧠 Analyze Heart Risk", use_container_width=True):

        # ================= 20 FEATURES (CORRECT ORDER) =================
        X = np.array([[ 
            age,                   # 1 Age
            gender,                # 2 Gender
            bp,                    # 3 Blood Pressure
            chol,                  # 4 Cholesterol Level
            1,                     # 5 Exercise Habits (Moderate)
            smoking,               # 6 Smoking
            0,                     # 7 Family Heart Disease
            diabetes,              # 8 Diabetes
            bmi,                   # 9 BMI
            0,                     # 10 High Blood Pressure
            0,                     # 11 Low HDL Cholesterol
            0,                     # 12 High LDL Cholesterol
            0,                     # 13 Alcohol Consumption
            stress,                # 14 Stress Level
            sleep,                 # 15 Sleep Hours
            0,                     # 16 Sugar Consumption
            150,                   # 17 Triglyceride Level
            90,                    # 18 Fasting Blood Sugar
            1.2,                   # 19 CRP Level
            10                     # 20 Homocysteine Level
        ]])

        # SAFETY CHECK
        assert X.shape[1] == model.n_features_in_

        prob = model.predict_proba(X)[0][1]

        st.subheader("📊 Risk Assessment")

        st.progress(int(prob * 100))
        st.metric("Heart Disease Probability", f"{prob*100:.2f}%")

        if prob >= 0.7:
            st.error("🔴 HIGH RISK")
            st.write("👉 Immediate medical consultation recommended.")
        elif prob >= 0.4:
            st.warning("🟠 MODERATE RISK")
            st.write("👉 Lifestyle improvement advised.")
        else:
            st.success("🟢 LOW RISK")
            st.write("👉 Maintain healthy habits.")

        st.caption("⚠️ AI-based prediction — not a medical diagnosis.")

# ================= FOOTER =================
st.markdown("---")
st.caption("© 2025 HeartCare AI | Developed by Kamalesh")
