# CareSight Diabetes and Hypertension Screening

This project is a Streamlit-based healthcare screening dashboard that uses two trained machine learning models:

- Diabetes prediction model: XGBoost model stored in `model_dibetest/dibetes_xgboost_best_model.pkl`
- Hypertension prediction model: Random Forest model stored in `model_hypertension/best_hypertension_model.pkl`

The application supports:

- User registration and login
- SQLite-based local data storage
- Patient profile storage for shared details
- Diabetes screening with live inputs
- Hypertension screening with live inputs
- Placeholder fingerprint action for future hardware integration
- Combined health report export in TXT and PDF format

## Project Structure

```text
Dibetes_and_hypertension_project/
|-- app.py
|-- requirements.txt
|-- README.md
|-- COMMANDS.md
|-- WORKFLOW.md
|-- .streamlit/
|   `-- config.toml
|-- model_dibetest/
|   `-- dibetes_xgboost_best_model.pkl
`-- model_hypertension/
    `-- best_hypertension_model.pkl
```

## Model Inputs Used In The Live App

The UI uses the input schema from the deployed model files, not only the original training notes.

### Diabetes model inputs

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- BMI
- DiabetesPedigreeFunction
- Age

### Hypertension model inputs

- Age
- Gender
- Medical_History
- Smoking
- BMI
- Sporting
- Systolic_BP
- Diastolic_BP

## Application Flow

1. The user registers with a username and password.
2. The user logs in using the saved credentials.
3. The user fills patient details: name, mobile number, address, age, and gender.
4. These details are stored in the local SQLite database file `health_app.db`.
5. The dashboard opens and reuses shared patient fields such as age and gender.
6. The user can run diabetes and hypertension predictions separately.
7. The user can generate a combined health report and download it as TXT or PDF.

## Setup And Run

See the full command list in `COMMANDS.md`.

Basic steps:

1. Create a virtual environment.
2. Activate the virtual environment.
3. Install the dependencies from `requirements.txt`.
4. Run the Streamlit app.

## SQLite Storage

The app creates a local SQLite database file named `health_app.db` automatically on first run.

Stored information includes:

- Username
- Password hash
- Full name
- Mobile number
- Address
- Age
- Gender
- Account creation timestamp

## Risk Output Logic

Both deployed models are binary classifiers internally. The app converts the positive-class probability into UI risk bands:

- Low risk: probability less than 0.40
- Medium risk: probability from 0.40 to less than 0.70
- High risk: probability 0.70 or above

## Fingerprint Placeholder

The fingerprint feature is currently a UI placeholder only. The interface shows a confirmation message for demo flow, but no hardware integration is connected yet.

## Documentation Files

- `COMMANDS.md`: setup and run commands
- `WORKFLOW.md`: full project workflow and user journey

## Future Improvements

- Real fingerprint hardware integration
- PDF report branding with hospital logo and richer formatting
- Prediction history storage in SQLite
- Admin dashboard for multiple patients
