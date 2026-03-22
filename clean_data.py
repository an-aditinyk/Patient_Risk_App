import pandas as pd
import numpy as np

def clean_uci_data(input_path, output_path):
    print("Starting Advanced Data Cleaning...")
    df = pd.read_csv(input_path)
    df.replace('?', np.nan, inplace=True)
    
    # 1. Target Simplification
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    # 2. FEATURE ENGINEERING: Creating the 'Medical Intensity' score
    # Converting strings to numeric for calculation
    df['time_in_hospital'] = pd.to_numeric(df['time_in_hospital'], errors='coerce')
    df['num_medications'] = pd.to_numeric(df['num_medications'], errors='coerce')
    df['num_lab_procedures'] = pd.to_numeric(df['num_lab_procedures'], errors='coerce')
    
    # New Feature: Intensity of Care (More meds/labs per day = Higher Risk)
    df['intensity_score'] = (df['num_medications'] + df['num_lab_procedures']) / (df['time_in_hospital'] + 1)
    
    # 3. Clean and Select
    cols = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_medications', 
        'number_emergency', 'number_inpatient', 'diag_1', 'intensity_score', 'readmitted']

    df = df[cols].dropna()

    # Age mapping
    age_map = {f'[{i*10}-{i*10+10})': i*10+5 for i in range(10)}
    df['age'] = df['age'].replace(age_map)

    # Diagnosis mapping
    def map_diag(code):
        try:
            val = float(code)
            if 390 <= val <= 459: return 'Circulatory'
            if 460 <= val <= 519: return 'Respiratory'
            if val == 250: return 'Diabetes'
            return 'Other'
        except: return 'Other'
    
    df['diag_1'] = df['diag_1'].apply(map_diag)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data with Intensity Score saved to {output_path}")

if __name__ == "__main__":
    clean_uci_data('data/diabetic_data.csv', 'data/cleaned_hospital_data.csv')