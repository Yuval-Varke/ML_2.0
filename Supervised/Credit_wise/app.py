"""
Streamlit frontend for the CreditWise Loan Approval prediction.
Connects to the FastAPI backend running at http://localhost:8000.
"""

import streamlit as st
import requests
import os

# --- API Configuration ---
# Fallback to localhost if no environment variable is set
DEFAULT_API_URL = "http://localhost:8000"
API_URL = os.environ.get("BACKEND_URL", DEFAULT_API_URL)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise – Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
)

# ─── Custom CSS for premium look ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.85;
    }
    
    .result-approved {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-top: 1rem;
    }
    .result-denied {
        background: linear-gradient(135deg, #e53935, #e35d5b);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-top: 1rem;
    }
    .result-text {
        font-size: 2rem;
        font-weight: 700;
    }
    .result-sub {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    div.stButton > button {
        background: linear-gradient(135deg, #0f3460, #1a1a2e);
        color: white;
        border: none;
        padding: 0.75rem 2.5rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(15, 52, 96, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown(
    """
    <div class="main-header">
        <h1>🏦 CreditWise</h1>
        <p>AI-Powered Loan Approval Prediction System</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar for Deployment Config ---
with st.sidebar:
    st.header("⚙️ Deployment Settings")
    API_URL = st.text_input("Backend API URL", value=API_URL)
    st.info("When deployed, this should point to your Render URL (e.g. https://your-app.onrender.com)")

# --- Check backend health ---
try:
    health = requests.get(f"{API_URL}/health", timeout=3)
    if health.status_code != 200:
        st.error("⚠️ Backend API is not responding. Please start the FastAPI server first.")
        st.info("Run: `python main.py` in a separate terminal.")
        st.stop()
except requests.exceptions.ConnectionError:
    st.error("⚠️ Cannot connect to the backend API at http://localhost:8000")
    st.info("Run: `python main.py` in a separate terminal, then refresh this page.")
    st.stop()

# ─── Input Form ──────────────────────────────────────────────────────────────
st.markdown("### 📝 Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Information**")
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1, step=1)
    education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

with col2:
    st.markdown("**Financial Information**")
    applicant_income = st.number_input(
        "Applicant Income (₹/month)", min_value=1000, max_value=100000, value=10000, step=500
    )
    coapplicant_income = st.number_input(
        "Co-applicant Income (₹/month)", min_value=0, max_value=100000, value=2000, step=500
    )
    credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=650, step=5)
    existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=1, step=1)
    savings = st.number_input("Savings (₹)", min_value=0, max_value=200000, value=5000, step=500)

with col3:
    st.markdown("**Loan & Employment**")
    loan_amount = st.number_input(
        "Loan Amount (₹)", min_value=1000, max_value=200000, value=15000, step=500
    )
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84])
    dti_ratio = st.slider("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    collateral_value = st.number_input(
        "Collateral Value (₹)", min_value=0, max_value=500000, value=20000, step=1000
    )
    loan_purpose = st.selectbox("Loan Purpose", ["Business", "Car", "Education", "Home", "Personal"])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
    employment_status = st.selectbox(
        "Employment Status", ["Freelancer", "Salaried", "Self-employed", "Unemployed"]
    )
    employer_category = st.selectbox(
        "Employer Category", ["Business", "Government", "MNC", "Private", "Unemployed"]
    )

# ─── Predict button ─────────────────────────────────────────────────────────
st.markdown("---")

if st.button("🔍 Predict Loan Approval"):
    payload = {
        "applicant_income": float(applicant_income),
        "coapplicant_income": float(coapplicant_income),
        "age": float(age),
        "dependents": float(dependents),
        "credit_score": float(credit_score),
        "existing_loans": float(existing_loans),
        "dti_ratio": float(dti_ratio),
        "savings": float(savings),
        "collateral_value": float(collateral_value),
        "loan_amount": float(loan_amount),
        "loan_term": float(loan_term),
        "education_level": education_level,
        "employment_status": employment_status,
        "marital_status": marital_status,
        "loan_purpose": loan_purpose,
        "property_area": property_area,
        "gender": gender,
        "employer_category": employer_category,
    }

    with st.spinner("Analyzing your application…"):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                confidence = result["probability"] * 100

                if result["approved"]:
                    st.markdown(
                        f"""
                        <div class="result-approved">
                            <div class="result-text">✅ Loan APPROVED</div>
                            <div class="result-sub">
                                Confidence: {confidence:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.balloons()
                else:
                    st.markdown(
                        f"""
                        <div class="result-denied">
                            <div class="result-text">❌ Loan DENIED</div>
                            <div class="result-sub">
                                Confidence: {confidence:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                error_detail = response.json().get("detail", "Unknown error")
                st.error(f"Prediction failed: {error_detail}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888; font-size: 0.85rem;'>"
    "Built with ❤️ using FastAPI & Streamlit | CreditWise v1.0"
    "</p>",
    unsafe_allow_html=True,
)
