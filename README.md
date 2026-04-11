<p align="center">
  <img src="hero-banner.png" alt="HospitAI hero banner" width="100%">
</p>

<h1 align="center">🏥 HospitAI — Patient Risk App</h1>

<p align="center">
  <strong>A machine learning application for predicting patient readmission risk and turning hospital data into actionable clinical insight.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/Joblib-Model%20Artifacts-informational?style=flat-square" alt="Joblib">
</p>

---

## Why this project exists

Hospital readmissions are costly, operationally disruptive, and often preventable.

**HospitAI** is built to support better discharge decisions by estimating the probability that a patient may be readmitted, using a machine learning workflow wrapped in an interactive Streamlit application. Instead of stopping at model training, this project carries the idea all the way into a usable decision-support interface.

Because a model hidden in a notebook is just a very expensive opinion.

---

## What the app does

The application is organized into three practical modules:

- **Clinical Predictor** — estimates patient readmission risk using core clinical and visit-history features
- **Data Intake (Log Outcome)** — appends new patient outcomes into the cleaned dataset for future retraining
- **Executive Analytics** — presents leadership-facing insights and visual metrics for impact analysis

This makes the project useful not only as an ML prototype, but as a full mini-product for healthcare analytics storytelling.

---

## Core features

- **Readmission risk prediction** using a trained Random Forest classifier
- **Custom feature engineering** with an Intensity Score:
  
  \[
  \text{Intensity Score} = \frac{\text{Medications} + \text{Lab Procedures}}{\text{Hospital Stay} + 1}
  \]

- **Interactive Streamlit interface** for real-time risk scoring
- **Outcome logging pipeline** to continuously grow the dataset
- **Executive dashboard** for analytics and communication
- **Saved inference artifacts** using `joblib`

---

## Pipeline overview

<p align="center">
  <img src="pipeline-arch.png" alt="Pipeline Architecture" width="92%">
</p>

The system follows a clean end-to-end flow:

```text
cleaned_hospital_data.csv
        ↓
Feature Engineering
        ↓
Manual Class Balancing
        ↓
Random Forest Training
        ↓
risk_model.pkl + model_cols.pkl
        ↓
Streamlit Clinical App
```

The training pipeline reads the cleaned hospital dataset, balances the readmission classes, trains a Random Forest model, and saves both the trained model and its expected feature columns for use inside the app.

---

## App preview

<p align="center">
  <img src="dashboard-overview.png" alt="HospitAI dashboard overview" width="92%">
</p>

The Streamlit interface supports a simple workflow:

1. Enter patient information.
2. Run the risk analysis.
3. Review the readmission probability.
4. Log the eventual outcome when available.
5. Use analytics for reporting and improvement.

Basically: predictive healthcare, but with fewer screenshots from Jupyter and more actual usability.

---

## Project structure

```bash
Patient_Risk_App/
│
├── app.py
├── train_model.py
├── utils.py
├── README.md
├── requirements.txt
│
├── data/
│   └── cleaned_hospital_data.csv
│
└── models/
    ├── risk_model.pkl
    └── model_cols.pkl
```

---

## How it works

### 1. Input collection
The app takes patient-level inputs such as age, days in hospital, number of lab procedures, number of medications, emergency visits, inpatient visits, and diagnosis category.

### 2. Feature engineering
It computes a custom **Intensity Score** to capture how resource-heavy the patient's care has been.

### 3. Prediction
The app aligns the processed input to the saved training columns and passes it into the model to generate a readmission probability.

### 4. Outcome logging
Users can append new labeled outcomes back into the cleaned dataset, which creates a simple feedback loop for future retraining.

That gives the project a much stronger story than “I trained a classifier and left it there.”

---

## Model training

The training script:

- loads `data/cleaned_hospital_data.csv`
- one-hot encodes the input features
- balances the dataset by downsampling the majority class
- trains a `RandomForestClassifier`
- saves:
  - `models/risk_model.pkl`
  - `models/model_cols.pkl`

Run training with:

```bash
python train_model.py
```

---

## Running the app

Start the Streamlit application with:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal, usually:

```bash
http://localhost:8501
```

---

## Tech stack

| Layer | Technology |
|---|---|
| App UI | Streamlit |
| Data processing | Pandas |
| ML model | Scikit-learn |
| Visualizations | Plotly |
| Model serialization | Joblib |
| Language | Python |

---

## Why this project stands out

A lot of student ML projects stop at:
- train model
- print accuracy
- disappear

This one goes further:
- builds a usable prediction interface
- adds feature engineering
- supports data logging
- includes executive-facing analytics
- creates a stronger real-world deployment narrative

That makes it much better as a portfolio project, internship project, or healthcare analytics showcase.

---

## Future improvements

Some strong next steps for the project:

- add model evaluation metrics directly in the app
- include SHAP or feature contribution explanations
- calibrate predicted probabilities
- support automated retraining
- deploy publicly on Streamlit Cloud
- add better input validation and audit logging
- create separate clinician and admin views

---

## Installation

Clone the repository:

```bash
git clone https://github.com/an-aditinyk/Patient_Risk_App.git
cd Patient_Risk_App
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If needed, install core packages manually:

```bash
pip install streamlit pandas scikit-learn plotly joblib
```

---

## Author

Built by **Aditi Nayak**.

If this project helped you, inspired you, or made your portfolio look significantly more employable, consider giving it a ⭐

---

## Final note

In healthcare, early intervention matters.

In GitHub, so does a README that doesn’t look like it was discharged too soon.
