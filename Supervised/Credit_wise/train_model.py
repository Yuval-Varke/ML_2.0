"""
Train and save the loan approval model along with all preprocessors.
This script reproduces the preprocessing from Credit_wise.ipynb
and saves the model artifacts for use by the FastAPI backend.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    # ─── 1. Load Data ─────────────────────────────────────────────────────────
    df = pd.read_csv("loan_approval_data.csv")
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ─── 2. Drop Applicant_ID (not a feature) ────────────────────────────────
    if "Applicant_ID" in df.columns:
        df = df.drop(columns=["Applicant_ID"])

    # ─── 3. Identify column types ────────────────────────────────────────────
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["float64"]).columns.tolist()

    # ─── 4. Handle missing values ────────────────────────────────────────────
    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

    # ─── 5. Encode Education_Level & Loan_Approved with LabelEncoder ─────────
    le_education = LabelEncoder()
    df["Education_Level"] = le_education.fit_transform(df["Education_Level"])

    le_target = LabelEncoder()
    df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])

    # ─── 6. OneHotEncode remaining categorical columns ───────────────────────
    ohe_cols = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category",
    ]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[ohe_cols])
    encoded_df = pd.DataFrame(
        encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index
    )
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    print(f"Processed dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # ─── 7. Split features / target ──────────────────────────────────────────
    X = df.drop(columns=["Loan_Approved"])
    y = df["Loan_Approved"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ─── 8. Train a RandomForest model ───────────────────────────────────────
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

    # ─── 9. Save all artifacts ───────────────────────────────────────────────
    artifacts = {
        "model": model,
        "num_imputer": num_imp,
        "cat_imputer": cat_imp,
        "label_encoder_education": le_education,
        "label_encoder_target": le_target,
        "onehot_encoder": ohe,
        "ohe_cols": ohe_cols,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names,
    }
    joblib.dump(artifacts, "model_artifacts.joblib")
    print("\n✅ Model artifacts saved to model_artifacts.joblib")


if __name__ == "__main__":
    main()
