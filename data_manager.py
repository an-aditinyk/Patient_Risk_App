import pandas as pd
import numpy as np
import os

DATA_FILE = 'hospital_data.csv'

def initialize_data():
    """Generates the initial dataset if it doesn't exist."""
    if not os.path.exists(DATA_FILE):
        np.random.seed(42)
        n = 1000
        data = {
            'Age': np.random.randint(18, 90, n),
            'Time_in_Hospital': np.random.randint(1, 14, n),
            'Lab_Procedures': np.random.randint(5, 100, n),
            'Prior_Visits': np.random.poisson(1.2, n),
            'Diagnosis': np.random.choice(['Diabetes', 'Heart Failure', 'Infection', 'Injury'], n)
        }
        df = pd.DataFrame(data)
        
        # LOGIC: Higher Risk = Older + Long Stay + Prior Visits
        # This creates the "pattern" for the AI to learn
        risk_score = (df['Age']*0.1 + df['Time_in_Hospital']*0.4 + df['Prior_Visits']*1.5)
        df['Readmitted'] = (risk_score > risk_score.quantile(0.85)).astype(int)
        
        df.to_csv(DATA_FILE, index=False)
        return "✅ Initial dataset 'hospital_data.csv' created."
    return "ℹ️ Dataset already exists."

def add_patient_record(age, stay, labs, prior, diag, readmitted):
    """Appends a new patient record to the CSV."""
    new_row = pd.DataFrame([[age, stay, labs, prior, diag, readmitted]], 
                           columns=['Age', 'Time_in_Hospital', 'Lab_Procedures', 'Prior_Visits', 'Diagnosis', 'Readmitted'])
    new_row.to_csv(DATA_FILE, mode='a', header=False, index=False)

if __name__ == "__main__":
    print(initialize_data())