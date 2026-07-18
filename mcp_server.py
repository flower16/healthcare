"""MCP server for the Hospital Readmission Risk Analytics project.

Exposes the readmission-risk model as Model Context Protocol tools so external
coordinators (e.g. the SolarBillIQ multi-agent router) can call it over stdio,
without launching the Streamlit UI. Reuses risk_core.py (the Streamlit-free logic).

Lives at the project root (NOT a folder named `mcp/`) so the `mcp` SDK package
isn't shadowed; chdir/path-fix so risk_core imports regardless of launch cwd.

Run:  python mcp_server.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from mcp.server.fastmcp import FastMCP  # noqa: E402

import risk_core  # noqa: E402

mcp = FastMCP("healthcare")

_state: dict = {}  # lazily trained model + held-out data, built once


def _ensure_model():
    if "model" not in _state:
        df = risk_core.generate_data()
        model, X_test, y_test, y_proba, auc = risk_core.train_model(df)
        _state.update(df=df, model=model, X_test=X_test, y_test=y_test, auc=auc)
    return _state


@mcp.tool()
def predict_readmission_risk(age: int, length_of_stay: int, comorbidities: int,
                             admission_type: str = "Emergency",
                             discharge_department: str = "Cardiology") -> dict:
    """Predict a patient's 30-day hospital-readmission risk.

    admission_type: Emergency | Elective | Urgent.
    discharge_department: Cardiology | Oncology | Orthopedics | General Surgery |
    Internal Medicine | Neurology. Returns probability + a plain-language risk tier.
    """
    s = _ensure_model()
    risk = risk_core.predict_risk(s["model"], {
        "age": age, "length_of_stay": length_of_stay, "comorbidities": comorbidities,
        "admission_type": admission_type, "discharge_department": discharge_department,
    })
    tier = ("low" if risk < 0.20 else "moderate" if risk < 0.50 else "high")
    return {"risk": round(risk, 4), "tier": tier,
            "overall_rate": round(float(s["df"]["Readmitted_30_Days"].mean()), 4)}


@mcp.tool()
def cohort_readmission_rates() -> dict:
    """Readmission rates overall and by admission type / discharge department."""
    s = _ensure_model()
    return {**risk_core.cohort_rates(s["df"]), "model_roc_auc": round(s["auc"], 4)}


@mcp.tool()
def model_drivers() -> dict:
    """Permutation feature importance — what drives readmission risk in the model."""
    s = _ensure_model()
    return {"model_roc_auc": round(s["auc"], 4),
            "drivers": risk_core.feature_importance(s["model"], s["X_test"], s["y_test"])}


if __name__ == "__main__":
    _ensure_model()   # train at startup (sync, fast) so the first tool call is instant
    mcp.run()         # stdio transport
