import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved pipeline
pipeline = joblib.load('heart_disease_model.pkl')

# Page config
st.set_page_config(page_title="Heart Disease Screening Tool", page_icon="🫀")
st.title("🫀 Heart Disease Risk Screening Tool")
st.write("Enter patient details to assess coronary artery disease risk.")
st.caption("Screening aid only — not a diagnostic confirmation.")

st.divider()

# Two columns for cleaner layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Demographics")
    age = st.number_input("Age (years)", min_value=20, max_value=100, value=54)
    sex = st.selectbox("Sex", options=[1, 0], 
                       format_func=lambda x: "Male" if x == 1 else "Female")
    
    st.subheader("Basic Vitals")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 
                                min_value=80, max_value=220, value=130,
                                help="Systolic BP measured on admission")
    chol = st.number_input("Serum Cholesterol (mg/dL)", 
                            min_value=100, max_value=600, value=246)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL",
                        options=[0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    st.subheader("Cardiac Symptoms")
    cp = st.selectbox("Chest Pain Type", 
                       options=[1, 2, 3, 4],
                       format_func=lambda x: {
                           1: "1 — Typical Angina",
                           2: "2 — Atypical Angina", 
                           3: "3 — Non-Anginal Pain",
                           4: "4 — Asymptomatic"
                       }[x])
    exang = st.selectbox("Exercise-Induced Angina",
                          options=[0, 1],
                          format_func=lambda x: "Yes" if x == 1 else "No")
    thalach = st.number_input("Max Heart Rate Achieved (bpm)", 
                               min_value=60, max_value=220, value=150,
                               help="During stress test. Expected max ≈ 220 − age")

st.divider()
st.subheader("Diagnostic Test Results")

col3, col4 = st.columns(2)

with col3:
    restecg = st.selectbox("Resting ECG Result",
                            options=[0, 1, 2],
                            format_func=lambda x: {
                                0: "0 — Normal",
                                1: "1 — ST-T Wave Abnormality",
                                2: "2 — Left Ventricular Hypertrophy"
                            }[x])
    oldpeak = st.number_input("ST Depression (Exercise vs Rest)", 
                               min_value=0.0, max_value=7.0, value=1.0, step=0.1,
                               help="Higher values indicate more ischaemia")
    slope = st.selectbox("ST Segment Slope (Peak Exercise)",
                          options=[1, 2, 3],
                          format_func=lambda x: {
                              1: "1 — Upsloping (normal)",
                              2: "2 — Flat (borderline)",
                              3: "3 — Downsloping (concerning)"
                          }[x])

with col4:
    ca = st.selectbox("Major Vessels with Blockage (Fluoroscopy)",
                       options=[0, 1, 2, 3],
                       format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''}")
    thal = st.selectbox("Nuclear Stress Test Result",
                         options=[3, 6, 7],
                         format_func=lambda x: {
                             3: "3 — Normal",
                             6: "6 — Fixed Defect (old infarct)",
                             7: "7 — Reversible Defect (active ischaemia)"
                         }[x])

st.divider()

if st.button("Assess Risk", type="primary"):
    # Build input dataframe with correct column names
    input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                               restecg, thalach, exang, oldpeak, slope, ca, thal]],
                             columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                      'restecg', 'thalach', 'exang', 'oldpeak', 
                                      'slope', 'ca', 'thal'])

    # Predict
    probability = pipeline.predict_proba(input_df)[0][1]
    THRESHOLD = 0.3
    risk = "HIGH RISK" if probability >= THRESHOLD else "LOW RISK"

    # Display result
    if probability >= THRESHOLD:
        st.error(f"### 🔴 Result: {risk}")
    else:
        st.success(f"### 🟢 Result: {risk}")

    st.metric("Probability of Heart Disease", f"{probability:.1%}")

    # Risk breakdown
    st.subheader("Risk Factors Summary")
    
    factors = []
    if ca >= 2:
        factors.append(f"⚠️ {ca} major vessel(s) with blockage — significant finding")
    if thal == 7:
        factors.append("⚠️ Reversible defect on stress test — active ischaemia")
    if cp == 4:
        factors.append("⚠️ Asymptomatic — silent ischaemia possible")
    if oldpeak >= 2.0:
        factors.append(f"⚠️ ST depression {oldpeak}mm — significant ischaemia marker")
    if exang == 1:
        factors.append("⚠️ Exercise-induced angina present")
    if thalach < (220 - age) * 0.75:
        factors.append(f"⚠️ Max HR {thalach} bpm — below 75% of expected maximum")

    if factors:
        for f in factors:
            st.write(f)
    else:
        st.write("✅ No major individual risk flags identified.")

    # Clinical recommendation
    st.divider()
    if probability >= THRESHOLD:
        st.warning("**Recommendation:** Refer for cardiology evaluation. "
                   "Consider stress ECG, echocardiogram, or coronary angiography "
                   "based on clinical judgement.")
    else:
        st.info("**Recommendation:** Low risk — routine follow-up. "
                "Reassess if symptoms develop.")

    st.caption(f"Threshold: {THRESHOLD} | Model: Random Forest Pipeline | "
               f"Dataset: UCI Cleveland (n=303)")import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved pipeline
pipeline = joblib.load('heart_disease_model.pkl')

# Page config
st.set_page_config(page_title="Heart Disease Screening Tool", page_icon="🫀")
st.title("🫀 Heart Disease Risk Screening Tool")
st.write("Enter patient details to assess coronary artery disease risk.")
st.caption("Screening aid only — not a diagnostic confirmation.")

st.divider()

# Two columns for cleaner layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Demographics")
    age = st.number_input("Age (years)", min_value=20, max_value=100, value=54)
    sex = st.selectbox("Sex", options=[1, 0], 
                       format_func=lambda x: "Male" if x == 1 else "Female")
    
    st.subheader("Basic Vitals")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 
                                min_value=80, max_value=220, value=130,
                                help="Systolic BP measured on admission")
    chol = st.number_input("Serum Cholesterol (mg/dL)", 
                            min_value=100, max_value=600, value=246)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL",
                        options=[0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    st.subheader("Cardiac Symptoms")
    cp = st.selectbox("Chest Pain Type", 
                       options=[1, 2, 3, 4],
                       format_func=lambda x: {
                           1: "1 — Typical Angina",
                           2: "2 — Atypical Angina", 
                           3: "3 — Non-Anginal Pain",
                           4: "4 — Asymptomatic"
                       }[x])
    exang = st.selectbox("Exercise-Induced Angina",
                          options=[0, 1],
                          format_func=lambda x: "Yes" if x == 1 else "No")
    thalach = st.number_input("Max Heart Rate Achieved (bpm)", 
                               min_value=60, max_value=220, value=150,
                               help="During stress test. Expected max ≈ 220 − age")

st.divider()
st.subheader("Diagnostic Test Results")

col3, col4 = st.columns(2)

with col3:
    restecg = st.selectbox("Resting ECG Result",
                            options=[0, 1, 2],
                            format_func=lambda x: {
                                0: "0 — Normal",
                                1: "1 — ST-T Wave Abnormality",
                                2: "2 — Left Ventricular Hypertrophy"
                            }[x])
    oldpeak = st.number_input("ST Depression (Exercise vs Rest)", 
                               min_value=0.0, max_value=7.0, value=1.0, step=0.1,
                               help="Higher values indicate more ischaemia")
    slope = st.selectbox("ST Segment Slope (Peak Exercise)",
                          options=[1, 2, 3],
                          format_func=lambda x: {
                              1: "1 — Upsloping (normal)",
                              2: "2 — Flat (borderline)",
                              3: "3 — Downsloping (concerning)"
                          }[x])

with col4:
    ca = st.selectbox("Major Vessels with Blockage (Fluoroscopy)",
                       options=[0, 1, 2, 3],
                       format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''}")
    thal = st.selectbox("Nuclear Stress Test Result",
                         options=[3, 6, 7],
                         format_func=lambda x: {
                             3: "3 — Normal",
                             6: "6 — Fixed Defect (old infarct)",
                             7: "7 — Reversible Defect (active ischaemia)"
                         }[x])

st.divider()

if st.button("Assess Risk", type="primary"):
    # Build input dataframe with correct column names
    input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                               restecg, thalach, exang, oldpeak, slope, ca, thal]],
                             columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                      'restecg', 'thalach', 'exang', 'oldpeak', 
                                      'slope', 'ca', 'thal'])

    # Predict
    probability = pipeline.predict_proba(input_df)[0][1]
    THRESHOLD = 0.3
    risk = "HIGH RISK" if probability >= THRESHOLD else "LOW RISK"

    # Display result
    if probability >= THRESHOLD:
        st.error(f"### 🔴 Result: {risk}")
    else:
        st.success(f"### 🟢 Result: {risk}")

    st.metric("Probability of Heart Disease", f"{probability:.1%}")

    # Risk breakdown
    st.subheader("Risk Factors Summary")
    
    factors = []
    if ca >= 2:
        factors.append(f"⚠️ {ca} major vessel(s) with blockage — significant finding")
    if thal == 7:
        factors.append("⚠️ Reversible defect on stress test — active ischaemia")
    if cp == 4:
        factors.append("⚠️ Asymptomatic — silent ischaemia possible")
    if oldpeak >= 2.0:
        factors.append(f"⚠️ ST depression {oldpeak}mm — significant ischaemia marker")
    if exang == 1:
        factors.append("⚠️ Exercise-induced angina present")
    if thalach < (220 - age) * 0.75:
        factors.append(f"⚠️ Max HR {thalach} bpm — below 75% of expected maximum")

    if factors:
        for f in factors:
            st.write(f)
    else:
        st.write("✅ No major individual risk flags identified.")

    # Clinical recommendation
    st.divider()
    if probability >= THRESHOLD:
        st.warning("**Recommendation:** Refer for cardiology evaluation. "
                   "Consider stress ECG, echocardiogram, or coronary angiography "
                   "based on clinical judgement.")
    else:
        st.info("**Recommendation:** Low risk — routine follow-up. "
                "Reassess if symptoms develop.")

    st.caption(f"Threshold: {THRESHOLD} | Model: Random Forest Pipeline | "
               f"Dataset: UCI Cleveland (n=303)")
