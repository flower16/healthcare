"""Streamlit-free core of the Hospital Readmission Risk model.

The Streamlit app (app.py) runs UI code at import time, so it can't be imported
from a non-Streamlit process. This module holds the same data-generation, training,
and prediction logic as plain functions (no `st.*`, no caching decorators) so the
MCP server — and any other consumer — can reuse it without launching Streamlit.

Logic mirrors app.py exactly; keep them in sync if the model changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ADMISSION_TYPES = ["Emergency", "Elective", "Urgent"]
DEPARTMENTS = ["Cardiology", "Oncology", "Orthopedics", "General Surgery",
               "Internal Medicine", "Neurology"]
NUMERIC_FEATURES = ["Age", "Length_of_Stay", "Number_of_Comorbidities"]
CATEGORICAL_FEATURES = ["Admission_Type", "Discharge_Department"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Readmitted_30_Days"


def generate_data(n_patients: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(loc=62, scale=17, size=n_patients).clip(18, 100).round()
    los = rng.gamma(shape=2.0, scale=2.5, size=n_patients).clip(1, 60).round()
    comorbidities = rng.poisson(lam=2.0, size=n_patients).clip(0, 10)
    admission_type = rng.choice(ADMISSION_TYPES, size=n_patients, p=[0.5, 0.3, 0.2])
    department = rng.choice(DEPARTMENTS, size=n_patients)

    emergency_effect = np.where(admission_type == "Emergency", 0.6, 0.0)
    elective_effect = np.where(admission_type == "Elective", -0.4, 0.0)
    dept_effect = np.select(
        [department == "Oncology", department == "Cardiology"], [0.5, 0.3], default=0.0)
    log_odds = (-3.2 + 0.025 * (age - 60) + 0.06 * los + 0.35 * comorbidities
                + emergency_effect + elective_effect + dept_effect)
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    readmitted = rng.binomial(1, prob).astype(bool)

    return pd.DataFrame({
        "Patient_ID": np.arange(1, n_patients + 1),
        "Age": age.astype(int), "Length_of_Stay": los.astype(int),
        "Admission_Type": admission_type,
        "Number_of_Comorbidities": comorbidities.astype(int),
        "Discharge_Department": department, "Readmitted_30_Days": readmitted,
    })


def train_model(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough")
    model = Pipeline([("preprocess", pre),
                      ("clf", RandomForestClassifier(
                          n_estimators=300, max_depth=8, min_samples_leaf=20,
                          class_weight="balanced", random_state=42, n_jobs=-1))])
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    return model, X_test, y_test, y_proba, auc


def predict_risk(model, patient: dict) -> float:
    row = pd.DataFrame([{
        "Age": patient["age"], "Length_of_Stay": patient["length_of_stay"],
        "Number_of_Comorbidities": patient["comorbidities"],
        "Admission_Type": patient["admission_type"],
        "Discharge_Department": patient["discharge_department"],
    }])
    return float(model.predict_proba(row)[:, 1][0])


def cohort_rates(df: pd.DataFrame) -> dict:
    return {
        "overall": float(df[TARGET].mean()),
        "by_admission_type": df.groupby("Admission_Type")[TARGET].mean().round(4).to_dict(),
        "by_department": df.groupby("Discharge_Department")[TARGET].mean().round(4).to_dict(),
    }


def feature_importance(model, X_test, y_test) -> list[dict]:
    r = permutation_importance(model, X_test, y_test, n_repeats=10,
                               random_state=42, n_jobs=-1)
    return (pd.DataFrame({"feature": X_test.columns,
                          "importance": r.importances_mean.round(4)})
            .sort_values("importance", ascending=False).to_dict("records"))
