import streamlit as st
import pandas as pd
import pickle

pipe = pickle.load(open("Stud.pkl", 'rb'))

st.title("🎓 Student Placement Prediction")

cgpa = st.number_input("CGPA", 0.0, 10.0, step=0.1)
internships = st.number_input("Internships", 0, 10)
projects = st.number_input("Projects", 0, 10)
workshops = st.number_input("Workshops/Certifications", 0, 10)
aptitude = st.number_input("Aptitude Test Score", 0, 100)
softskills = st.number_input("Soft Skills Rating", 0.0, 5.0, step=0.1)
ssc = st.number_input("SSC Marks", 0, 100)
hsc = st.number_input("HSC Marks", 0, 100)

extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
training = st.selectbox("Placement Training", ["Yes", "No"])

if st.button("Predict Placement Status"):

    input_df = pd.DataFrame({
        "CGPA": [cgpa],
        "Internships": [internships],
        "Projects": [projects],
        "Workshops/Certifications": [workshops],
        "AptitudeTestScore": [aptitude],
        "SoftSkillsRating": [softskills],
        "SSC_Marks": [ssc],
        "HSC_Marks": [hsc],
        "ExtracurricularActivities": [extra],
        "PlacementTraining": [training]
    })

   prediction = pipe.predict(input_df)[0]


if hasattr(pipe.named_steps['model'], "predict_proba"):
    probability = pipe.predict_proba(input_df)[0]
    st.success(f"Prediction: {prediction}")
    st.write(f"Confidence: {max(probability)*100:.2f}%")
else:
    st.success(f"Prediction: {prediction}")
    st.write("Confidence score not available for this model.")

