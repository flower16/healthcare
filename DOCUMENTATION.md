# Hospital Readmission Risk Analytics — Technical Documentation

This document describes the design, data model, and internals of the
application. For installation and run instructions, see the
[README](README.md).

---

## 1. Overview

The application is a single-file Streamlit app ([app.py](app.py)) that:

1. Generates a synthetic cohort of 10,000 patient discharges.
2. Runs exploratory data analysis (EDA) with interactive Plotly charts.
3. Trains an interpretable Random Forest classifier to predict whether a
   patient is readmitted within 30 days.
4. Reports the factors driving readmission risk via permutation importance.
5. Exposes a live **What-If** calculator that scores a hypothetical patient.

Everything runs locally; there is no database, external API, or persisted
state. Data generation and model training are cached, so the heavy work runs
once per session and the UI stays responsive.

> ⚠️ **The dataset is fully synthetic.** It is not derived from real patients
> and must not be used for any clinical decision.

---

## 2. Architecture

The app is organised top-to-bottom in a single module. There is no package
structure — the flow is linear and matches the order Streamlit executes the
script on each rerun.

```
app.py
├── Page configuration            st.set_page_config(...)
├── Module constants              ADMISSION_TYPES, DEPARTMENTS, FEATURES, ...
├── generate_data()               @st.cache_data    → synthetic DataFrame
├── train_model()                 @st.cache_resource → fitted pipeline + test split
├── compute_feature_importance()  @st.cache_data    → permutation importance table
├── Load + train (run once)       df = generate_data(); model, ... = train_model(df)
├── Sidebar                       What-If sliders → single-row patient frame → risk
└── Main page
    ├── Headline metrics          4 st.metric cards
    └── Tabs
        ├── 📊 Exploratory Analysis   rate-by-cohort bars, correlation heatmap, raw preview
        ├── 🤖 Model & Drivers        permutation importance bar, ROC curve
        └── 🧪 What-If Detail         risk gauge, risk tier, current profile
```

### Execution model

Streamlit reruns the whole script on every widget interaction. Caching is what
makes this cheap:

| Function | Decorator | Why |
| --- | --- | --- |
| `generate_data` | `@st.cache_data` | Returns a DataFrame (serializable); regenerated only if args change. |
| `train_model` | `@st.cache_resource` | Returns a fitted model + arrays (a non-serializable resource shared across reruns). |
| `compute_feature_importance` | `@st.cache_data` | Returns a small DataFrame; the `_model` arg is prefixed with `_` so Streamlit does not try to hash the estimator. |

Because of these caches, moving a sidebar slider only re-runs the cheap parts
(building the one-row patient frame, a single `predict_proba`, and re-rendering
charts) — not data generation or training.

---

## 3. Data dictionary

`generate_data(n_patients=10_000, seed=42)` returns a DataFrame with one row
per discharge:

| Column | Type | Description | Distribution |
| --- | --- | --- | --- |
| `Patient_ID` | int | Unique identifier, `1..n` | sequential |
| `Age` | int | Patient age in years | Normal(μ=62, σ=17), clipped to [18, 100] |
| `Length_of_Stay` | int | Inpatient days | Gamma(shape=2.0, scale=2.5), clipped to [1, 60] (right-skewed) |
| `Admission_Type` | str | `Emergency` / `Elective` / `Urgent` | p = [0.5, 0.3, 0.2] |
| `Number_of_Comorbidities` | int | Count of comorbid conditions | Poisson(λ=2.0), clipped to [0, 10] |
| `Discharge_Department` | str | One of six departments | uniform |
| `Readmitted_30_Days` | bool | Target: readmitted within 30 days | drawn from the latent risk model (below) |

Departments: Cardiology, Oncology, Orthopedics, General Surgery,
Internal Medicine, Neurology.

### Latent risk model

The target is **not** random noise. Each discharge has a true log-odds of
readmission built from its features, so the classifier has a genuine signal to
recover:

```
log_odds = -3.2
         + 0.025 * (Age - 60)          # older patients slightly higher risk
         + 0.06  * Length_of_Stay      # longer stays higher risk
         + 0.35  * Number_of_Comorbidities
         + (+0.6 if Emergency else 0)  # emergency admissions higher risk
         + (-0.4 if Elective  else 0)  # elective admissions lower risk
         + (+0.5 if Oncology, +0.3 if Cardiology, else 0)

prob       = sigmoid(log_odds)
Readmitted = Bernoulli(prob)
```

The intercept `-3.2` sets the base rate; the overall readmission rate lands in
a realistic single-digit-to-teens percentage range. `seed=42` makes the whole
dataset reproducible.

---

## 4. Modeling

`train_model(df)` builds a scikit-learn `Pipeline` and returns
`(model, X_test, y_test, y_proba, auc)`.

### Features and target

```python
NUMERIC_FEATURES     = ["Age", "Length_of_Stay", "Number_of_Comorbidities"]
CATEGORICAL_FEATURES = ["Admission_Type", "Discharge_Department"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET   = "Readmitted_30_Days"
```

Note `Patient_ID` is deliberately excluded — it carries no signal.

### Pipeline

```
Pipeline
├── preprocess: ColumnTransformer
│   ├── cat: OneHotEncoder(handle_unknown="ignore")  → CATEGORICAL_FEATURES
│   └── remainder="passthrough"                       → numeric features unchanged
└── clf: RandomForestClassifier
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
```

Key choices:

- **One-hot encoding inside the pipeline** means raw DataFrames (including the
  single-row What-If frame) can be passed straight to `predict_proba` without
  any manual encoding.
- **`handle_unknown="ignore"`** keeps prediction robust to category values not
  seen during training.
- **`class_weight="balanced"`** counteracts the class imbalance (readmission is
  the minority class).
- **`max_depth=8` / `min_samples_leaf=20`** keep the forest shallow and
  regularized — interpretable and resistant to overfitting rather than
  maximally accurate.

### Train/test split

`train_test_split(test_size=0.25, random_state=42, stratify=y)` — 75/25 split,
stratified on the target to preserve the readmission rate in both partitions.

### Evaluation

ROC AUC is computed on the held-out test set from `predict_proba[:, 1]` and
displayed both as a headline metric and as a full ROC curve in the
**Model & Drivers** tab.

### Feature importance

`compute_feature_importance` uses **permutation importance** on the test set
(`n_repeats=10`). It is computed on the original, un-encoded feature columns,
so the chart shows one bar per real-world factor (e.g. "Admission_Type") rather
than per one-hot dummy — keeping it interpretable for a clinical audience.
Error bars show the standard deviation across repeats.

---

## 5. User interface

### Headline metrics (always visible)

| Metric | Source |
| --- | --- |
| Total Discharges | `len(df)` |
| Overall Readmission Rate | `df[TARGET].mean()` |
| Model ROC AUC | `auc` from `train_model` |
| What-If Patient Risk | live `predict_proba` for the sidebar profile, with delta vs. the average rate (`delta_color="inverse"` → higher risk shown red) |

### Tabs

- **📊 Exploratory Analysis** — readmission rate by admission type (vertical
  bars) and by department (horizontal, sorted, red color scale); a correlation
  heatmap over Length of Stay, Comorbidities, Age and the outcome; and an
  expander previewing the first 100 raw rows.
- **🤖 Model & Drivers** — permutation importance bar chart and the ROC curve
  (model line vs. the random-classifier diagonal).
- **🧪 What-If Detail** — a Plotly gauge of the predicted risk (0–100%) with
  green/amber/red bands, a plain-language risk tier, and the current patient
  profile as a one-row table.

### What-If risk tiers

| Predicted risk | Tier | Message |
| --- | --- | --- |
| `< 20%` | 🟢 Low | standard discharge follow-up |
| `20%–50%` | 🟡 Moderate | consider enhanced follow-up |
| `≥ 50%` | 🔴 High | recommend care-management intervention |

The sidebar sliders default to a 65-year-old, 5-day stay, 2 comorbidities.

---

## 6. Configuration & extension points

Most behavior is controlled by the module-level constants near the top of
[app.py](app.py):

- **Cohort size / reproducibility** — change `n_patients` or `seed` in
  `generate_data`.
- **Categories** — edit `ADMISSION_TYPES` and `DEPARTMENTS`. They feed both
  data generation and the sidebar selectboxes, so the UI stays in sync
  automatically.
- **Risk relationships** — adjust the coefficients in the latent `log_odds`
  expression to change how strongly each factor drives readmission.
- **Model** — swap the `RandomForestClassifier` for another estimator (e.g.
  `LogisticRegression`) inside the pipeline; nothing else needs to change
  because preprocessing and importance are model-agnostic.
- **Risk-tier thresholds** — edit the `if risk < 0.20 / 0.50` cutoffs in the
  What-If Detail tab.

---

## 7. Dependencies

From [requirements.txt](requirements.txt):

| Package | Min version | Used for |
| --- | --- | --- |
| pandas | 2.0 | DataFrame construction and aggregation |
| numpy | 1.24 | Random sampling and the latent risk model |
| scikit-learn | 1.3 | Pipeline, Random Forest, permutation importance, metrics |
| streamlit | 1.30 | Web UI, caching, layout |
| plotly | 5.18 | Interactive charts and the risk gauge |

---

## 8. Limitations

- **Synthetic data only.** Results reflect the hand-coded latent model, not
  real clinical outcomes. Do not use for patient care.
- **No persistence.** State lives in memory for the session; nothing is saved.
- **AUC reflects recoverable signal**, not real-world predictive power — the
  data is generated from a known logistic model the forest is approximating.
- **No hyperparameter tuning or cross-validation** — the model is intentionally
  simple and interpretable rather than optimized.
