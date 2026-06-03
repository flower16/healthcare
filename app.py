"""
Hospital Readmission Risk Analytics
===================================
A Streamlit application that analyzes 30-day hospital readmission trends to
identify high-risk patient cohorts. It generates synthetic discharge data,
performs interactive EDA, trains an interpretable ML model, surfaces the
factors driving readmission risk, and offers a live "What-If" risk calculator.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hospital Readmission Risk Analytics",
    page_icon="🏥",
    layout="wide",
)

# Constants used across data generation and the model
ADMISSION_TYPES = ["Emergency", "Elective", "Urgent"]
DEPARTMENTS = [
    "Cardiology",
    "Oncology",
    "Orthopedics",
    "General Surgery",
    "Internal Medicine",
    "Neurology",
]
NUMERIC_FEATURES = ["Age", "Length_of_Stay", "Number_of_Comorbidities"]
CATEGORICAL_FEATURES = ["Admission_Type", "Discharge_Department"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Readmitted_30_Days"


# ---------------------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------------------
@st.cache_data
def generate_data(n_patients: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic discharge dataset of ``n_patients`` rows.

    Readmission is generated from a latent logistic model so that the
    probability of readmission depends realistically on age, length of stay,
    comorbidities and admission type. This gives the ML model a genuine
    signal to learn rather than pure noise.
    """
    rng = np.random.default_rng(seed)

    # --- Patient demographics & clinical features -------------------------
    age = rng.normal(loc=62, scale=17, size=n_patients).clip(18, 100).round()
    # Length of stay is right-skewed (most stays are short).
    los = rng.gamma(shape=2.0, scale=2.5, size=n_patients).clip(1, 60).round()
    comorbidities = rng.poisson(lam=2.0, size=n_patients).clip(0, 10)

    admission_type = rng.choice(ADMISSION_TYPES, size=n_patients, p=[0.5, 0.3, 0.2])
    department = rng.choice(DEPARTMENTS, size=n_patients)

    # --- Latent risk model ------------------------------------------------
    # Each factor contributes to the log-odds of readmission.
    emergency_effect = np.where(admission_type == "Emergency", 0.6, 0.0)
    elective_effect = np.where(admission_type == "Elective", -0.4, 0.0)
    # Oncology and Cardiology patients carry slightly higher baseline risk.
    dept_effect = np.select(
        [department == "Oncology", department == "Cardiology"],
        [0.5, 0.3],
        default=0.0,
    )

    log_odds = (
        -3.2
        + 0.025 * (age - 60)
        + 0.06 * los
        + 0.35 * comorbidities
        + emergency_effect
        + elective_effect
        + dept_effect
    )
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    readmitted = rng.binomial(1, prob).astype(bool)

    df = pd.DataFrame(
        {
            "Patient_ID": np.arange(1, n_patients + 1),
            "Age": age.astype(int),
            "Length_of_Stay": los.astype(int),
            "Admission_Type": admission_type,
            "Number_of_Comorbidities": comorbidities.astype(int),
            "Discharge_Department": department,
            "Readmitted_30_Days": readmitted,
        }
    )
    return df


# ---------------------------------------------------------------------------
# 3. Predictive model training
# ---------------------------------------------------------------------------
@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train a Random Forest pipeline to predict 30-day readmission.

    Returns the fitted pipeline plus held-out test data and the ROC AUC so the
    UI can report performance. Categorical features are one-hot encoded inside
    the pipeline, so raw DataFrames can be passed straight to ``predict``.
    """
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",  # numeric features pass through unchanged
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    return model, X_test, y_test, y_proba, auc


@st.cache_data
def compute_feature_importance(_model, X_test: pd.DataFrame, y_test: pd.Series):
    """Permutation importance on the held-out set.

    Permutation importance is model-agnostic and measured on the original
    (un-encoded) feature columns, so the chart stays interpretable for
    clinicians — one bar per real-world factor rather than per dummy variable.
    The leading underscore on ``_model`` tells Streamlit not to hash it.
    """
    result = permutation_importance(
        _model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    importance = (
        pd.DataFrame(
            {
                "Feature": X_test.columns,
                "Importance": result.importances_mean,
                "Std": result.importances_std,
            }
        )
        .sort_values("Importance", ascending=True)
        .reset_index(drop=True)
    )
    return importance


# ---------------------------------------------------------------------------
# Load data and train the model (cached, so this runs once per session)
# ---------------------------------------------------------------------------
df = generate_data()
model, X_test, y_test, y_proba, auc = train_model(df)

# ===========================================================================
# Sidebar — Interactive "What-If" controls (5)
# ===========================================================================
st.sidebar.header("🧪 What-If Risk Calculator")
st.sidebar.caption(
    "Adjust a hypothetical patient's profile to see their predicted "
    "30-day readmission risk update in real time."
)

age_input = st.sidebar.slider("Age", 18, 100, 65)
los_input = st.sidebar.slider("Length of Stay (days)", 1, 60, 5)
comorbidities_input = st.sidebar.slider("Number of Comorbidities", 0, 10, 2)
admission_input = st.sidebar.selectbox("Admission Type", ADMISSION_TYPES)
department_input = st.sidebar.selectbox("Discharge Department", DEPARTMENTS)

# Build a single-row DataFrame with the same schema the model expects.
patient = pd.DataFrame(
    [
        {
            "Age": age_input,
            "Length_of_Stay": los_input,
            "Number_of_Comorbidities": comorbidities_input,
            "Admission_Type": admission_input,
            "Discharge_Department": department_input,
        }
    ]
)
risk = float(model.predict_proba(patient)[:, 1][0])

# ===========================================================================
# Main page
# ===========================================================================
st.title("🏥 Hospital Readmission Risk Analytics")
st.markdown(
    "Identify **high-risk patient cohorts** for 30-day readmission using "
    "synthetic discharge data, interactive analytics, and an interpretable "
    "machine-learning model."
)

# --- Headline metrics ------------------------------------------------------
overall_rate = df[TARGET].mean()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Discharges", f"{len(df):,}")
col2.metric("Overall Readmission Rate", f"{overall_rate:.1%}")
col3.metric("Model ROC AUC", f"{auc:.3f}")
col4.metric(
    "What-If Patient Risk",
    f"{risk:.1%}",
    delta=f"{(risk - overall_rate):+.1%} vs. avg",
    delta_color="inverse",  # higher risk shown in red
)

st.divider()

# Organise the content into tabs for a clean layout.
tab_eda, tab_model, tab_whatif = st.tabs(
    ["📊 Exploratory Analysis", "🤖 Model & Drivers", "🧪 What-If Detail"]
)

# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------------------
with tab_eda:
    st.subheader("Readmission Rates by Cohort")

    left, right = st.columns(2)

    # Readmission rate by admission type
    with left:
        rate_by_admission = (
            df.groupby("Admission_Type")[TARGET].mean().reset_index()
        )
        fig = px.bar(
            rate_by_admission,
            x="Admission_Type",
            y=TARGET,
            color="Admission_Type",
            text_auto=".1%",
            title="Readmission Rate by Admission Type",
            labels={TARGET: "Readmission Rate"},
        )
        fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig, width="stretch")

    # Readmission rate by discharge department
    with right:
        rate_by_dept = (
            df.groupby("Discharge_Department")[TARGET]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig = px.bar(
            rate_by_dept,
            x=TARGET,
            y="Discharge_Department",
            orientation="h",
            color=TARGET,
            color_continuous_scale="Reds",
            text_auto=".1%",
            title="Readmission Rate by Discharge Department",
            labels={TARGET: "Readmission Rate"},
        )
        fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Correlation Matrix")
    st.caption(
        "Correlation between Length of Stay, Number of Comorbidities, Age and "
        "the readmission outcome."
    )
    corr = df[
        ["Length_of_Stay", "Number_of_Comorbidities", "Age"]
    ].assign(Readmitted_30_Days=df[TARGET].astype(int)).corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Feature Correlation Heatmap",
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("Preview raw synthetic data"):
        st.dataframe(df.head(100), width="stretch")

# ---------------------------------------------------------------------------
# 4. Feature importance & model performance
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader("What Drives Readmission Risk?")
    importance = compute_feature_importance(model, X_test, y_test)

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        error_x="Std",
        color="Importance",
        color_continuous_scale="Viridis",
        title="Permutation Feature Importance",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Higher values mean shuffling that feature degrades the model more — "
        "i.e. the feature carries more predictive signal for readmission."
    )

    st.subheader("Model Performance (ROC Curve)")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Model (AUC = {auc:.3f})")
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Random",
        )
    )
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="Receiver Operating Characteristic",
    )
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# 5. What-If detail view
# ---------------------------------------------------------------------------
with tab_whatif:
    st.subheader("Predicted Readmission Risk for This Patient")
    st.markdown(
        "Use the **sidebar controls** to change the patient profile. "
        "The gauge below updates instantly."
    )

    # Risk gauge
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk * 100,
            number={"suffix": "%"},
            title={"text": "30-Day Readmission Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred"},
                "steps": [
                    {"range": [0, 20], "color": "#d4edda"},
                    {"range": [20, 50], "color": "#fff3cd"},
                    {"range": [50, 100], "color": "#f8d7da"},
                ],
            },
        )
    )
    st.plotly_chart(fig, width="stretch")

    # Plain-language risk tier
    if risk < 0.20:
        st.success("🟢 **Low risk** — standard discharge follow-up.")
    elif risk < 0.50:
        st.warning("🟡 **Moderate risk** — consider enhanced follow-up.")
    else:
        st.error("🔴 **High risk** — recommend care-management intervention.")

    st.markdown("**Current patient profile:**")
    st.dataframe(patient, width="stretch", hide_index=True)

st.divider()
st.caption(
    "⚠️ Data is fully synthetic and for demonstration only. Not for clinical use."
)
