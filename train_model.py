import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib
import os

def train_balanced_model():
    df = pd.read_csv('data/cleaned_hospital_data.csv')
    X = pd.get_dummies(df.drop('readmitted', axis=1))
    y = df['readmitted']

    # 1. Manually Balance the Dataset (Under-sampling the majority)
    df_majority = df[df.readmitted == 0]
    df_minority = df[df.readmitted == 1]
    
    # Downsample majority to match minority count
    df_majority_downsampled = resample(df_majority, replace=False, 
                                     n_samples=len(df_minority), random_state=42)
    
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    X_bal = pd.get_dummies(df_balanced.drop('readmitted', axis=1))
    y_bal = df_balanced.readmitted

    # 2. Train Model
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # 3. Save
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(model, 'models/risk_model.pkl')
    joblib.dump(X_bal.columns.tolist(), 'models/model_cols.pkl')
    print("Balanced Model Trained & Saved!")

if __name__ == "__main__":
    train_balanced_model()