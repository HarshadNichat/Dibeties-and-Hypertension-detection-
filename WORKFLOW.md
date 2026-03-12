# Project Workflow

## 1. Project Purpose

This project provides a modern Streamlit user interface for screening diabetes and hypertension using already trained machine learning models.

## 2. Main Workflow

### Step 1: Registration

- A new user opens the app.
- The user creates a username and password.
- The account is stored in the SQLite database.

### Step 2: Login

- The user logs in using the registered username and password.
- The system verifies the password hash from SQLite.

### Step 3: Patient Details Entry

- After login, the app opens the patient registration page.
- The user enters:
  - Full name
  - Mobile number
  - Address
  - Age
  - Gender
- These details are saved in SQLite.

### Step 4: Dashboard Access

- Once the profile is complete, the user enters the main dashboard.
- The dashboard shows guidance and patient summary cards.

### Step 5: Diabetes Screening

- The user opens the diabetes section.
- The app asks for the deployed model inputs:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - BMI
  - DiabetesPedigreeFunction
- Age is reused automatically from the saved profile.
- The user can click the fingerprint placeholder button.
- The app runs the XGBoost model and displays:
  - Prediction label
  - Probability
  - Low, Medium, or High risk band

### Step 6: Hypertension Screening

- The user opens the hypertension section.
- The app asks for the deployed model inputs:
  - Medical_History
  - Smoking
  - BMI
  - Sporting
  - Systolic_BP
  - Diastolic_BP
- Age and gender are reused automatically from the saved profile.
- The user can click the fingerprint placeholder button.
- The app runs the Random Forest model and displays:
  - Prediction label
  - Probability
  - Low, Medium, or High risk band

### Step 7: Health Report Generation

- The user clicks the generate report button.
- The app builds a combined patient report.
- The report includes:
  - Name
  - Mobile number
  - Address
  - Age
  - Gender
  - Username
  - Diabetes result
  - Hypertension result
- The report can be downloaded as:
  - TXT file
  - PDF file

## 3. Data Storage Workflow

- SQLite file name: `health_app.db`
- Table used: `users`
- Authentication and patient profile are stored in the same table for this version of the app.

## 4. Model Workflow

### Diabetes model

- File: `model_dibetest/dibetes_xgboost_best_model.pkl`
- Algorithm: XGBoost classifier

### Hypertension model

- File: `model_hypertension/best_hypertension_model.pkl`
- Algorithm: Random Forest classifier

## 5. Future Workflow Extensions

- Replace the fingerprint placeholder with a real hardware reader
- Add patient history records
- Add PDF layout improvements with branding
- Add multi-patient management
