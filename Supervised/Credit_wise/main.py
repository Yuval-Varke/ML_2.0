"""
FastAPI backend for Loan Approval prediction.
Loads the trained model artifacts and exposes /predict endpoint.
"""

import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ─── Load model artifacts ─────────────────────────────────────────────────────
ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "model_artifacts.joblib")

if not os.path.exists(ARTIFACT_PATH):
    raise FileNotFoundError(
        f"Model artifacts not found at {ARTIFACT_PATH}. "
        "Run  python train_model.py  first."
    )

artifacts = joblib.load(ARTIFACT_PATH)
model = artifacts["model"]
num_imputer = artifacts["num_imputer"]
label_encoder_education = artifacts["label_encoder_education"]
label_encoder_target = artifacts["label_encoder_target"]
onehot_encoder = artifacts["onehot_encoder"]
ohe_cols = artifacts["ohe_cols"]
feature_names = artifacts["feature_names"]

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="CreditWise – Loan Approval API",
    description="Predict whether a loan application will be approved.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ──────────────────────────────────────────────
class LoanApplication(BaseModel):
    """Input schema – raw applicant fields (before encoding)."""

    applicant_income: float = Field(..., gt=0, description="Applicant monthly income")
    coapplicant_income: float = Field(..., ge=0, description="Co-applicant monthly income")
    age: float = Field(..., ge=18, le=100, description="Applicant age")
    dependents: float = Field(..., ge=0, description="Number of dependents")
    credit_score: float = Field(..., ge=300, le=900, description="Credit score (300-900)")
    existing_loans: float = Field(..., ge=0, description="Number of existing loans")
    dti_ratio: float = Field(..., ge=0, le=1, description="Debt-to-Income ratio (0-1)")
    savings: float = Field(..., ge=0, description="Total savings")
    collateral_value: float = Field(..., ge=0, description="Collateral asset value")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_term: float = Field(..., gt=0, description="Loan term in months (12-84)")
    education_level: str = Field(
        ..., description="Education level: 'Graduate' or 'Not Graduate'"
    )
    employment_status: str = Field(
        ...,
        description="Employment status: 'Freelancer', 'Salaried', 'Self-employed', or 'Unemployed'",
    )
    marital_status: str = Field(
        ..., description="Marital status: 'Married' or 'Single'"
    )
    loan_purpose: str = Field(
        ...,
        description="Purpose: 'Business', 'Car', 'Education', 'Home', or 'Personal'",
    )
    property_area: str = Field(
        ..., description="Property area: 'Rural', 'Semiurban', or 'Urban'"
    )
    gender: str = Field(..., description="Gender: 'Female' or 'Male'")
    employer_category: str = Field(
        ...,
        description="Employer: 'Business', 'Government', 'MNC', 'Private', or 'Unemployed'",
    )


class PredictionResponse(BaseModel):
    approved: bool
    label: str
    probability: float


# ─── Helper: preprocess a single application ─────────────────────────────────
def preprocess(app_data: LoanApplication) -> pd.DataFrame:
    """Apply the same transformations used during training."""

    # Build a raw DataFrame that mirrors the original CSV (after Applicant_ID drop)
    raw = pd.DataFrame(
        [
            {
                "Applicant_Income": app_data.applicant_income,
                "Coapplicant_Income": app_data.coapplicant_income,
                "Age": app_data.age,
                "Dependents": app_data.dependents,
                "Credit_Score": app_data.credit_score,
                "Existing_Loans": app_data.existing_loans,
                "DTI_Ratio": app_data.dti_ratio,
                "Savings": app_data.savings,
                "Collateral_Value": app_data.collateral_value,
                "Loan_Amount": app_data.loan_amount,
                "Loan_Term": app_data.loan_term,
                "Education_Level": app_data.education_level,
                "Employment_Status": app_data.employment_status,
                "Marital_Status": app_data.marital_status,
                "Loan_Purpose": app_data.loan_purpose,
                "Property_Area": app_data.property_area,
                "Gender": app_data.gender,
                "Employer_Category": app_data.employer_category,
            }
        ]
    )

    # Label-encode Education_Level
    raw["Education_Level"] = label_encoder_education.transform(
        raw["Education_Level"]
    )

    # One-hot encode the categorical columns
    encoded = onehot_encoder.transform(raw[ohe_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=onehot_encoder.get_feature_names_out(ohe_cols),
        index=raw.index,
    )
    processed = pd.concat([raw.drop(columns=ohe_cols), encoded_df], axis=1)

    # Reorder columns to match training
    processed = processed[feature_names]
    return processed


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "CreditWise Loan Approval API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    """Return a loan approval prediction for the given application."""
    try:
        X = preprocess(application)
        proba = model.predict_proba(X)[0]  # [P(No), P(Yes)]
        pred_idx = int(np.argmax(proba))
        label = label_encoder_target.inverse_transform([pred_idx])[0]
        return PredictionResponse(
            approved=(label == "Yes"),
            label=label,
            probability=float(proba[pred_idx]),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
