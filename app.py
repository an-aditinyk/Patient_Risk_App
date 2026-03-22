import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# --- 1. SETTINGS & ASSETS ---
# Removed custom CSS to ensure text visibility in all themes
st.set_page_config(page_title="HospitAI | Clinical Suite", layout="wide")

@st.cache_resource
def load_assets():
    # Ensure these paths match your folder structure
    model = joblib.load('models/risk_model.pkl')
    cols = joblib.load('models/model_cols.pkl')
    return model, cols

def save_new_data(age, stay, labs, meds, emergency, inpatient, diag, readmitted):
    """Appends new patient data to the cleaned CSV file with exactly 9 columns."""
    outcome = 1 if readmitted == "Yes" else 0
    # Calculate Intensity Score
    intensity = (meds + labs) / (stay + 1)
    
    # We MUST match these 9 columns exactly as they appear in cleaned_hospital_data.csv
    # age, time_in_hospital, num_lab_procedures, num_medications, 
    # number_emergency, number_inpatient, diag_1, intensity_score, readmitted
    
    new_entry = pd.DataFrame([[
        age, stay, labs, meds, emergency, inpatient, diag, intensity, outcome
    ]], columns=['age', 'time_in_hospital', 'num_lab_procedures', 'num_medications', 
                 'number_emergency', 'number_inpatient', 'diag_1', 'intensity_score', 'readmitted'])
    
    # Append to CSV
    new_entry.to_csv('data/cleaned_hospital_data.csv', mode='a', header=False, index=False)


# Load Brain
try:
    model, model_cols = load_assets()
except Exception as e:
    st.error(f"AI Brain not found. Error: {e}")
    st.stop()

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("Patient Risk Suite")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Clinical Predictor", "Data Intake (Log Outcome)", "Executive Analytics"])

# --- PAGE 1: CLINICAL PREDICTOR ---
if page == "Clinical Predictor":
    st.title("Patient Readmission Risk Analysis")
    st.markdown("Predictive Discharge tool to identify the $26 Billion Leak.")
    
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        st.subheader("Patient Vitals & History")
        age = st.slider("Patient Age", 18, 100, 65)
        stay = st.slider("Days in Hospital", 1, 30, 5)
        labs = st.number_input("Total Lab Procedures", 1, 200, 45)
        meds = st.number_input("Number of Medications", 1, 100, 18)
        prior_e = st.number_input("Emergency Visits (Past Year)", 0, 20, 0)
        prior_i = st.number_input("Inpatient Visits (Past Year)", 0, 20, 0)
        diag = st.selectbox("Primary Diagnosis Category", ["Circulatory", "Respiratory", "Diabetes", "Other"])
        st.info("The **Intensity Score** is calculated as: (Meds + Labs) / (Stay + 1). It measures how 'resource-heavy' a patient's care is per day.")
        run_analysis = st.button("Run AI Risk Analysis", use_container_width=True)

    if run_analysis:
        # Background Feature Engineering
        intensity = (meds + labs) / (stay + 1)
        
        input_dict = {
            'age': age, 'time_in_hospital': stay, 'num_lab_procedures': labs,
            'num_medications': meds, 'number_emergency': prior_e, 
            'number_inpatient': prior_i, 'intensity_score': intensity
        }
        for d in ["Circulatory", "Respiratory", "Diabetes", "Other"]:
            input_dict[f"diag_1_{d}"] = 1 if diag == d else 0
        
        input_df = pd.DataFrame([input_dict])[model_cols]
        prob = model.predict_proba(input_df)[0][1]
        
        with c2:
            st.subheader("Risk Analysis & Protocol")
            # Metric will now use system-default colors
            st.metric("Probability of Readmission", f"{prob:.1%}")
            
            if prob > 0.65:
                st.error("CRITICAL: HIGH RISK INTERVENTION")
                st.write("**Recommended Protocol:**")
                st.write("- Home Visit: RN visit required within 48 hours.")
                st.write("- Med-Rec: Pharmacist review of all home medications.")
                st.write("- Bridge Appt: PCP follow-up confirmed within 5 days.")
            elif prob > 0.40:
                st.warning("CAUTION: MEDIUM RISK MONITORING")
                st.write("**Recommended Protocol:**")
                st.write("- Telehealth: Care Coordinator call on Day 3.")
                st.write("- Social Check: Verify transportation and meal access.")
            else:
                st.success("STABLE: LOW RISK PROTOCOL")
                st.write("**Recommended Protocol:**")
                st.write("- Standard Discharge: Provide printed recovery guide.")
                st.write("- PCP Sync: Electronic summary sent to primary doctor.")

# --- PAGE 2: DATA INTAKE ---
elif page == "Data Intake (Log Outcome)":
    st.title("Clinical Outcome Logging")
    st.info("Record actual patient outcomes to maintain model accuracy.")
    
    with st.form("intake_form", clear_on_submit=True):
        f_c1, f_c2 = st.columns(2)
        with f_c1:
            f_age = st.number_input("Age", 18, 100, 65)
            f_stay = st.number_input("Stay Duration (Days)", 1, 30, 5)
            f_labs = st.number_input("Lab Count", 1, 200, 45)
        with f_c2:
            f_meds = st.number_input("Medication Count", 1, 100, 15)
            f_diag = st.selectbox("Diagnosis", ["Circulatory", "Respiratory", "Diabetes", "Other"])
            f_readm = st.radio("Was patient readmitted within 30 days?", ["No", "Yes"])
        
        if st.form_submit_button("Save Patient Record"):
            save_new_data(f_age, f_stay, f_labs, f_meds, 0, 0, f_diag, f_readm)
            st.success(f"Record for Patient (Age {f_age}) saved to database.")

    st.divider()
    st.subheader("Recent Hospital Records")
    if os.path.exists('data/cleaned_hospital_data.csv'):
        history_df = pd.read_csv('data/cleaned_hospital_data.csv')
        st.dataframe(history_df.tail(5), use_container_width=True)
# --- PAGE 3: EXECUTIVE ANALYTICS ---
elif page == "Executive Analytics":
    st.title("Executive ROI Dashboard")
    st.markdown("Financial impact and clinical distribution of readmission risks.")
    
    if os.path.exists('data/cleaned_hospital_data.csv'):
        df_dash = pd.read_csv('data/cleaned_hospital_data.csv')
        
        # 1. Financial metrics
        total_patients = len(df_dash)
        readmits = df_dash[df_dash['readmitted'] == 1].shape[0]
        potential_loss = readmits * 15200
        projected_savings = potential_loss * 0.25 
        
        # Display KPI Cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records", total_patients)
        m2.metric("Revenue Leak", f"${potential_loss:,}")
        m3.metric("Target Savings", f"${projected_savings:,}", delta="25% Goal")
        
        st.divider()
        
        # 2. Interactive Charts (The Fix is here)
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Risk by Diagnosis")
            # We use a professional color sequence for the specialties
            fig_diag = px.bar(df_dash, 
                             x='diag_1', 
                             y='readmitted', 
                             color='diag_1', 
                             title="Total Readmission Cases by Specialty",
                             labels={'diag_1': 'Medical Specialty', 'readmitted': 'Count of Readmits'},
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            
            fig_diag.update_layout(showlegend=False) 
            st.plotly_chart(fig_diag, use_container_width=True)
            
        with col_right:
            st.subheader("Age vs. Readmission Status")
            # Transform 0/1 to labels to fix the legend and colors
            df_dash['Status'] = df_dash['readmitted'].map({0: 'Stable', 1: 'Readmitted'})
            
            fig_age = px.histogram(df_dash, 
                                  x="age", 
                                  color="Status", 
                                  nbins=10, 
                                  title="Age Distribution of Risk",
                                  # Green for Stable, Red for Readmitted
                                  color_discrete_map={"Stable": "#2ecc71", "Readmitted": "#e74c3c"},
                                  category_orders={"Status": ["Stable", "Readmitted"]})
            
            st.plotly_chart(fig_age, use_container_width=True)
            
    else:
        st.error("No data found. Please run clean_data.py or add records in Data Intake.")