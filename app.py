from __future__ import annotations

import hashlib
import io
import sqlite3
import textwrap
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "health_app.db"
DIABETES_MODEL_PATH = BASE_DIR / "model_dibetest" / "dibetes_xgboost_best_model.pkl"
HYPERTENSION_MODEL_PATH = BASE_DIR / "model_hypertension" / "best_hypertension_model.pkl"

DIABETES_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

HYPERTENSION_FEATURES = [
    "Age",
    "Gender",
    "Medical_History",
    "Smoking",
    "BMI",
    "Sporting",
    "Systolic_BP",
    "Diastolic_BP",
]

GENDER_OPTIONS = ["Male", "Female"]


# Configure the Streamlit page before any UI is rendered.
def setup_page() -> None:
    st.set_page_config(
        page_title="CareSight Health Screening",
        page_icon="+",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# Inject the custom hospital-style theme and component styling.
def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --primary: #0b7285;
                --primary-dark: #075985;
                --accent: #2f855a;
                --warning: #b7791f;
                --danger: #c53030;
                --bg-soft: #f4fbfc;
                --surface: rgba(255, 255, 255, 0.94);
                --text-main: #16324f;
                --text-muted: #587085;
                --border: rgba(11, 114, 133, 0.14);
            }

            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(11, 114, 133, 0.10), transparent 30%),
                    radial-gradient(circle at left center, rgba(47, 133, 90, 0.08), transparent 28%),
                    linear-gradient(180deg, #eef8fb 0%, #f8fcfd 48%, #ffffff 100%);
                color: var(--text-main);
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            .hero-card,
            .info-card,
            .result-card,
            .metric-card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 22px;
                box-shadow: 0 24px 60px rgba(15, 35, 56, 0.08);
                padding: 1.4rem;
            }

            .hero-card {
                padding: 2rem;
                background: linear-gradient(135deg, rgba(7, 89, 133, 0.95), rgba(11, 114, 133, 0.92));
                color: #ffffff;
            }

            .hero-title {
                font-size: 2.1rem;
                font-weight: 700;
                margin: 0 0 0.6rem 0;
            }

            .hero-copy,
            .muted-copy {
                color: rgba(255, 255, 255, 0.88);
                font-size: 1rem;
                line-height: 1.6;
            }

            .info-card h4,
            .metric-card h4,
            .result-card h4 {
                margin: 0 0 0.5rem 0;
                color: var(--text-main);
            }

            .metric-label {
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: var(--text-muted);
            }

            .metric-value {
                font-size: 1.35rem;
                font-weight: 700;
                color: var(--text-main);
                margin-top: 0.2rem;
            }

            .risk-low {
                border-left: 6px solid var(--accent);
            }

            .risk-medium {
                border-left: 6px solid var(--warning);
            }

            .risk-high {
                border-left: 6px solid var(--danger);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f7fcfd 0%, #edf7fa 100%);
                border-right: 1px solid rgba(11, 114, 133, 0.12);
            }

            div[data-testid="stExpander"] {
                background: rgba(255, 255, 255, 0.84);
                border: 1px solid var(--border);
                border-radius: 18px;
                overflow: hidden;
            }

            .section-title {
                font-size: 1.4rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
                color: var(--text-main);
            }

            .section-copy {
                color: var(--text-muted);
                margin-bottom: 1rem;
            }

            .stButton > button,
            .stDownloadButton > button {
                border-radius: 999px;
                padding: 0.68rem 1.2rem;
                border: none;
                font-weight: 600;
                background: linear-gradient(135deg, var(--primary-dark), var(--primary));
                color: #ffffff;
            }

            .stButton > button:hover,
            .stDownloadButton > button:hover {
                filter: brightness(1.04);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
# Load and cache the deployed machine learning models from disk.
def load_models() -> dict[str, object]:
    return {
        "diabetes": joblib.load(DIABETES_MODEL_PATH),
        "hypertension": joblib.load(HYPERTENSION_MODEL_PATH),
    }


# Create the SQLite schema used for authentication and patient profile storage.
def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                mobile TEXT,
                address TEXT,
                age INTEGER,
                gender TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


# Hash plain text passwords before storing or comparing them.
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# Insert a new user account into SQLite during registration.
def create_user(username: str, password: str) -> tuple[bool, str]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO users (username, password_hash, created_at)
                VALUES (?, ?, ?)
                """,
                (username.strip(), hash_password(password), datetime.now().isoformat(timespec="seconds")),
            )
        return True, "Registration completed. Please log in."
    except sqlite3.IntegrityError:
        return False, "This username already exists. Choose a different username."


# Validate login credentials and return the matching user row when valid.
def authenticate_user(username: str, password: str) -> sqlite3.Row | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM users WHERE username = ? AND password_hash = ?",
            (username.strip(), hash_password(password)),
        ).fetchone()


    # Fetch the latest stored data for the currently logged-in user.
def get_user(user_id: int) -> sqlite3.Row | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


    # Save or update the shared patient details used across both screenings.
def update_user_profile(
    user_id: int,
    full_name: str,
    mobile: str,
    address: str,
    age: int,
    gender: str,
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE users
            SET full_name = ?, mobile = ?, address = ?, age = ?, gender = ?
            WHERE id = ?
            """,
            (full_name.strip(), mobile.strip(), address.strip(), age, gender, user_id),
        )


# Check whether the user already filled the required patient information.
def profile_complete(user: sqlite3.Row) -> bool:
    required_fields = [user["full_name"], user["mobile"], user["address"], user["age"], user["gender"]]
    return all(value not in (None, "") for value in required_fields)


# Initialize all Streamlit session keys used across the application flow.
def init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "user_id": None,
        "current_view": "auth",
        "diabetes_result": None,
        "hypertension_result": None,
        "diabetes_fp_ready": False,
        "hypertension_fp_ready": False,
        "report_text": None,
        "report_pdf": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


    # Store the authenticated session and reset previous screening state after login.
def login_user(user: sqlite3.Row) -> None:
    st.session_state.authenticated = True
    st.session_state.user_id = user["id"]
    st.session_state.diabetes_result = None
    st.session_state.hypertension_result = None
    st.session_state.diabetes_fp_ready = False
    st.session_state.hypertension_fp_ready = False
    st.session_state.report_text = None
    st.session_state.report_pdf = None
    st.session_state.current_view = "dashboard" if profile_complete(user) else "profile"


# Clear the active session and return the interface to the login screen.
def logout_user() -> None:
    for key in [
        "authenticated",
        "user_id",
        "current_view",
        "diabetes_result",
        "hypertension_result",
        "diabetes_fp_ready",
        "hypertension_fp_ready",
        "report_text",
        "report_pdf",
    ]:
        st.session_state[key] = None if key in {"user_id", "diabetes_result", "hypertension_result", "report_text", "report_pdf"} else False
    st.session_state.current_view = "auth"
    st.rerun()


# Convert model probability into the UI-friendly risk bands.
def risk_band(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


# Map each risk band to the matching CSS class for styled result cards.
def risk_css_class(risk: str) -> str:
    return {
        "Low": "risk-low",
        "Medium": "risk-medium",
        "High": "risk-high",
    }[risk]


# Convert the stored gender value into the numeric format used by the model.
def gender_to_model_value(gender: str) -> int:
    return 1 if gender == "Male" else 0


# Convert simple Yes or No dropdowns into the numeric format used by the model.
def yes_no_to_int(value: str) -> int:
    return 1 if value == "Yes" else 0


# Render a reusable styled information card using HTML markup.
def render_html_card(title: str, body: str, card_class: str = "info-card") -> None:
    st.markdown(
        f"""
        <div class="{card_class}">
            <h4>{title}</h4>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Show the combined login and registration entry page.
def render_auth_page() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">CareSight Diabetes and Hypertension Screening</div>
            <div class="hero-copy">
                Register, log in, store patient details in SQLite, and run live screening against the deployed
                diabetes and hypertension models from this project.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        st.markdown('<div class="section-title">Welcome back</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Use your registered username and password to access the screening dashboard.</div>',
            unsafe_allow_html=True,
        )
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", use_container_width=True, key="login_button"):
            if not username.strip() or not password:
                st.error("Username and password are required.")
            else:
                user = authenticate_user(username, password)
                if user is None:
                    st.error("Invalid username or password.")
                else:
                    login_user(user)
                    st.rerun()

    with register_tab:
        st.markdown('<div class="section-title">Create your account</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Register once, then log in later with the same username and password.</div>',
            unsafe_allow_html=True,
        )
        username = st.text_input("Choose a username", key="register_username")
        password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm password", type="password", key="register_confirm_password")
        if st.button("Register", use_container_width=True, key="register_button"):
            if len(username.strip()) < 4:
                st.error("Username must be at least 4 characters.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = create_user(username, password)
                if success:
                    st.success(message)
                else:
                    st.error(message)


# Render the sidebar navigation and account actions after login.
def render_sidebar(user: sqlite3.Row) -> None:
    with st.sidebar:
        st.markdown("### CareSight Panel")
        st.write(f"Logged in as: {user['username']}")
        st.write(f"Profile status: {'Complete' if profile_complete(user) else 'Pending'}")

        if st.button("Patient Details", use_container_width=True):
            st.session_state.current_view = "profile"
        if st.button("Screening Dashboard", use_container_width=True, disabled=not profile_complete(user)):
            st.session_state.current_view = "dashboard"
        if st.button("Log Out", use_container_width=True):
            logout_user()


# Capture and save the patient details shared by both prediction forms.
def render_profile_page(user: sqlite3.Row) -> None:
    st.markdown('<div class="section-title">Patient Registration</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Store the patient name, mobile number, address, age, and gender in SQLite before starting the screenings.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        full_name = st.text_input("Full name", value=user["full_name"] or "")
        mobile = st.text_input("Mobile number", value=user["mobile"] or "")
        address = st.text_area("Address", value=user["address"] or "", height=120)
        form_col1, form_col2 = st.columns(2)
        age = form_col1.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=int(user["age"]) if user["age"] else 30,
            step=1,
        )
        gender = form_col2.selectbox(
            "Gender",
            GENDER_OPTIONS,
            index=GENDER_OPTIONS.index(user["gender"]) if user["gender"] in GENDER_OPTIONS else 0,
        )

        if st.button("Save Patient Details", use_container_width=True):
            if not full_name.strip() or not mobile.strip() or not address.strip():
                st.error("Name, mobile number, and address are required.")
            else:
                update_user_profile(user["id"], full_name, mobile, address, int(age), gender)
                st.session_state.current_view = "dashboard"
                st.success("Patient details saved successfully.")
                st.rerun()

    with col2:
        render_html_card(
            "Why this page comes first",
            "<p>The screening forms reuse age and gender directly from the stored patient profile so the user does not need to enter the same information again.</p>",
        )
        st.write("")
        render_html_card(
            "Stored in SQLite",
            "<p>This app creates a local SQLite database file named health_app.db in the project folder and stores the patient details against the registered user account.</p>",
        )


# Show the dashboard hero section, saved profile data, and usage guidance.
def render_patient_summary(user: sqlite3.Row) -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Diabetes and Hypertension Detection</div>
            <div class="hero-copy">
                Common patient details are already synced from the saved profile. Enter the live measurements below,
                use the placeholder fingerprint step, and run the model predictions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    summary_columns = st.columns(5)
    summary_pairs = [
        ("Patient Name", user["full_name"]),
        ("Mobile", user["mobile"]),
        ("Age", user["age"]),
        ("Gender", user["gender"]),
        ("Address", user["address"]),
    ]

    for column, (label, value) in zip(summary_columns, summary_pairs):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    guidance_col1, guidance_col2 = st.columns(2)
    with guidance_col1:
        render_html_card(
            "User Guidance",
            "<p>1. Open one screening section at a time.</p><p>2. Fill the live clinical values carefully.</p><p>3. The fingerprint step is a placeholder and will be connected to hardware later.</p>",
        )
    with guidance_col2:
        render_html_card(
            "Model Inputs Used",
            "<p>Diabetes model: 7 deployed inputs from the saved XGBoost model.</p><p>Hypertension model: 8 deployed inputs from the saved Random Forest model.</p><p>Risk bands are derived from the model probability to show Low, Medium, and High screening risk.</p>",
        )


# Display one prediction result card with risk band and probability details.
def render_result(result: dict[str, object], title: str) -> None:
    risk = str(result["risk_band"])
    probability_percent = f"{float(result['probability']) * 100:.2f}%"
    render_html_card(
        title,
        (
            f"<p><strong>Model output:</strong> {result['prediction_text']}</p>"
            f"<p><strong>Risk band:</strong> {risk}</p>"
            f"<p><strong>Positive probability:</strong> {probability_percent}</p>"
            f"<p><strong>Generated:</strong> {result['generated_at']}</p>"
        ),
        card_class=f"result-card {risk_css_class(risk)}",
    )


# Render the diabetes form, placeholder fingerprint step, and live prediction flow.
def render_diabetes_section(user: sqlite3.Row) -> None:
    with st.expander("Diabetes Screening", expanded=True):
        st.write("Provide the live diabetes screening measurements below. Age is reused from the stored profile.")
        col1, col2 = st.columns(2)

        pregnancies = col1.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = col2.number_input("Glucose", min_value=0, max_value=400, value=110, step=1)
        blood_pressure = col1.number_input("Blood Pressure", min_value=0, max_value=250, value=72, step=1)
        skin_thickness = col2.number_input("Skin Thickness", min_value=0, max_value=120, value=20, step=1)
        bmi = col1.number_input("BMI", min_value=0.0, max_value=100.0, value=26.5, step=0.1, format="%.1f")
        dpf = col2.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=5.0,
            value=0.45,
            step=0.01,
            format="%.2f",
        )
        st.text_input("Age from profile", value=str(user["age"]), disabled=True)

        if st.button("Use Fingerprint Hardware", key="diabetes_fingerprint"):
            st.session_state.diabetes_fp_ready = True
            st.info("Fingerprint integration is coming soon. Demo placeholder recorded for now.")

        if st.session_state.diabetes_fp_ready:
            st.success("Fingerprint status: confirmed in demo mode. Hardware integration is coming soon.")

        if st.button("Get Diabetes Prediction", use_container_width=True, key="diabetes_predict"):
            model = load_models()["diabetes"]
            payload = pd.DataFrame(
                [
                    {
                        "Pregnancies": int(pregnancies),
                        "Glucose": float(glucose),
                        "BloodPressure": float(blood_pressure),
                        "SkinThickness": float(skin_thickness),
                        "BMI": float(bmi),
                        "DiabetesPedigreeFunction": float(dpf),
                        "Age": int(user["age"]),
                    }
                ],
                columns=DIABETES_FEATURES,
            )
            probability = float(model.predict_proba(payload)[0][1])
            prediction = int(model.predict(payload)[0])
            risk = risk_band(probability)
            st.session_state.diabetes_result = {
                "prediction": prediction,
                "prediction_text": "Diabetes risk detected" if prediction == 1 else "No diabetes risk detected",
                "risk_band": risk,
                "probability": probability,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        if st.session_state.diabetes_result:
            render_result(st.session_state.diabetes_result, "Diabetes Screening Result")


# Render the hypertension form, placeholder fingerprint step, and live prediction flow.
def render_hypertension_section(user: sqlite3.Row) -> None:
    with st.expander("Hypertension Screening", expanded=False):
        st.write("Provide the live hypertension screening measurements below. Age and gender are reused from the stored profile.")
        col1, col2 = st.columns(2)

        st.text_input("Age from profile", value=str(user["age"]), disabled=True, key="hypertension_age")
        st.text_input("Gender from profile", value=str(user["gender"]), disabled=True, key="hypertension_gender")
        medical_history = col1.selectbox("Relevant medical history", ["No", "Yes"])
        smoking = col2.selectbox("Smoking", ["No", "Yes"])
        bmi = col1.number_input("BMI", min_value=0.0, max_value=100.0, value=24.8, step=0.1, format="%.1f", key="ht_bmi")
        sporting = col2.selectbox("Regular sporting activity", ["No", "Yes"])
        systolic = col1.number_input("Systolic BP", min_value=50, max_value=260, value=120, step=1)
        diastolic = col2.number_input("Diastolic BP", min_value=30, max_value=180, value=80, step=1)

        if st.button("Use Fingerprint Hardware", key="hypertension_fingerprint"):
            st.session_state.hypertension_fp_ready = True
            st.info("Fingerprint integration is coming soon. Demo placeholder recorded for now.")

        if st.session_state.hypertension_fp_ready:
            st.success("Fingerprint status: confirmed in demo mode. Hardware integration is coming soon.")

        if st.button("Get Hypertension Prediction", use_container_width=True, key="hypertension_predict"):
            model = load_models()["hypertension"]
            payload = pd.DataFrame(
                [
                    {
                        "Age": int(user["age"]),
                        "Gender": gender_to_model_value(str(user["gender"])),
                        "Medical_History": yes_no_to_int(medical_history),
                        "Smoking": yes_no_to_int(smoking),
                        "BMI": float(bmi),
                        "Sporting": yes_no_to_int(sporting),
                        "Systolic_BP": float(systolic),
                        "Diastolic_BP": float(diastolic),
                    }
                ],
                columns=HYPERTENSION_FEATURES,
            )
            probability = float(model.predict_proba(payload)[0][1])
            prediction = int(model.predict(payload)[0])
            risk = risk_band(probability)
            st.session_state.hypertension_result = {
                "prediction": prediction,
                "prediction_text": "Hypertension risk detected" if prediction == 1 else "No hypertension risk detected",
                "risk_band": risk,
                "probability": probability,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        if st.session_state.hypertension_result:
            render_result(st.session_state.hypertension_result, "Hypertension Screening Result")


# Build the plain text report that combines patient details and screening outputs.
def build_report(user: sqlite3.Row) -> str:
    diabetes = st.session_state.diabetes_result
    hypertension = st.session_state.hypertension_result

    diabetes_summary = (
        f"Status: {diabetes['prediction_text']}\n"
        f"Risk Band: {diabetes['risk_band']}\n"
        f"Positive Probability: {float(diabetes['probability']) * 100:.2f}%\n"
        f"Generated At: {diabetes['generated_at']}"
        if diabetes
        else "Diabetes screening has not been generated yet."
    )

    hypertension_summary = (
        f"Status: {hypertension['prediction_text']}\n"
        f"Risk Band: {hypertension['risk_band']}\n"
        f"Positive Probability: {float(hypertension['probability']) * 100:.2f}%\n"
        f"Generated At: {hypertension['generated_at']}"
        if hypertension
        else "Hypertension screening has not been generated yet."
    )

    return (
        "CareSight Health Report\n"
        "=======================\n\n"
        f"Name: {user['full_name']}\n"
        f"Mobile Number: {user['mobile']}\n"
        f"Address: {user['address']}\n"
        f"Age: {user['age']}\n"
        f"Gender: {user['gender']}\n"
        f"Username: {user['username']}\n"
        f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        "Diabetes Screening\n"
        "-------------------\n"
        f"{diabetes_summary}\n\n"
        "Hypertension Screening\n"
        "-----------------------\n"
        f"{hypertension_summary}\n"
    )


# Convert the generated text report into a downloadable PDF document.
def build_report_pdf(report_text: str) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    text_object = pdf.beginText(50, height - 50)
    text_object.setFont("Helvetica", 11)
    text_object.setLeading(16)

    max_chars_per_line = 92
    for raw_line in report_text.splitlines():
        wrapped_lines = textwrap.wrap(raw_line, width=max_chars_per_line) or [""]
        for line in wrapped_lines:
            if text_object.getY() <= 50:
                pdf.drawText(text_object)
                pdf.showPage()
                text_object = pdf.beginText(50, height - 50)
                text_object.setFont("Helvetica", 11)
                text_object.setLeading(16)
            text_object.textLine(line)

    pdf.drawText(text_object)
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


# Provide preview and export options for the generated health report.
def render_report_section(user: sqlite3.Row) -> None:
    st.write("")
    st.markdown('<div class="section-title">Health Report</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Generate a combined report with patient information and the latest prediction results.</div>',
        unsafe_allow_html=True,
    )

    if st.button("Generate Health Report", use_container_width=True, key="generate_report"):
        st.session_state.report_text = build_report(user)
        st.session_state.report_pdf = build_report_pdf(st.session_state.report_text)

    if st.session_state.report_text:
        st.text_area("Report Preview", value=st.session_state.report_text, height=280)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        text_name = f"health_report_{user['username']}_{timestamp}.txt"
        pdf_name = f"health_report_{user['username']}_{timestamp}.pdf"
        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "Download TXT Report",
                st.session_state.report_text,
                file_name=text_name,
                mime="text/plain",
                use_container_width=True,
            )
        with download_col2:
            st.download_button(
                "Download PDF Report",
                st.session_state.report_pdf,
                file_name=pdf_name,
                mime="application/pdf",
                use_container_width=True,
            )


# Render the full screening dashboard after the patient profile is complete.
def render_dashboard(user: sqlite3.Row) -> None:
    render_patient_summary(user)
    st.write("")
    render_diabetes_section(user)
    st.write("")
    render_hypertension_section(user)
    render_report_section(user)


# Run the full application flow and choose the correct page for the current session.
def main() -> None:
    setup_page()
    inject_styles()
    init_db()
    init_session_state()

    if not DIABETES_MODEL_PATH.exists() or not HYPERTENSION_MODEL_PATH.exists():
        st.error("Model files are missing. Place both model files in the configured project folders.")
        st.stop()

    if not st.session_state.authenticated or st.session_state.user_id is None:
        render_auth_page()
        return

    user = get_user(int(st.session_state.user_id))
    if user is None:
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.current_view = "auth"
        st.warning("Your session expired. Please log in again.")
        render_auth_page()
        return

    render_sidebar(user)

    if st.session_state.current_view == "profile" or not profile_complete(user):
        render_profile_page(user)
    else:
        render_dashboard(user)


if __name__ == "__main__":
    main()