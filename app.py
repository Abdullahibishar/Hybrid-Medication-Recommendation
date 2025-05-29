# app.py
import streamlit as st
import pandas as pd
from pipeline import HybridRecommenderPipeline

# Custom CSS for styling the app
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        text-align: center;
        color: #2c3e50;
        font-size: 3rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Medication & Disease Prediction System")
st.write("This application provides personalized medication recommendations and a likely disease prediction based on the symptoms you enter.")

# Sidebar instructions for non-technical users
st.sidebar.header("About This App")
st.sidebar.info(
    """
    **How It Works:**
    - **Symptom Input:** Enter the patient's symptoms.
    - The system compares these symptoms to historical cases.
    - It predicts the most common disease among similar cases.
    - It then recommends medications based on those cases.
    
    **Count:** Indicates how many similar cases suggested that medication.
    """
)

# Cache the pipeline resource so it loads only once.
@st.cache_resource
def load_pipeline(csv_path):
    return HybridRecommenderPipeline(csv_path)

# Load the pipeline (ensure the CSV path is correct)
pipeline = load_pipeline("/Users/abdullahi/Documents/MedicationRecommender/personalized_medication_dataset.csv")

# Cache disease prediction and medication recommendations based on symptoms
@st.cache_data
def get_disease_and_recommendations(symptoms_text, top_n=5):
    return pipeline.get_recommendations_from_symptoms(symptoms_text, top_n=top_n)

# User input: Symptoms as free text
symptoms_input = st.text_area("Enter the patient's symptoms (e.g., fever, cough, headache):", "")

if st.button("Get Predictions"):
    if symptoms_input.strip() == "":
        st.error("Please enter some symptoms.")
    else:
        predicted_disease, med_recommendations = get_disease_and_recommendations(symptoms_input, top_n=5)
        
        st.markdown("### Predicted Disease")
        st.write(f"The system predicts the most likely disease is: **{predicted_disease}**")
        
        st.markdown("### Top 5 Medication Recommendations")
        st.dataframe(med_recommendations)
        
        st.markdown("### How to Interpret These Results")
        st.write(
            """
            - **Predicted Disease:** The most common diagnosis among similar cases.
            - **Medication Recommendations:** The number next to each medication shows how many similar cases suggested that medication.
            - A higher count means there is stronger agreement among similar cases.
            """
        )
