# 🏥 Hospital Readmission Risk Analytics

A Streamlit web app that analyzes 30-day hospital readmission trends to identify
high-risk patient cohorts. It generates synthetic discharge data, runs
interactive EDA, trains an interpretable ML model, shows the factors driving
readmission risk, and includes a live **What-If** risk calculator.

## Features

1. **Synthetic data generation** — 10,000 patient discharges with `Patient_ID`,
   `Age`, `Length_of_Stay`, `Admission_Type`, `Number_of_Comorbidities`,
   `Discharge_Department`, and `Readmitted_30_Days`. Readmission is drawn from a
   latent logistic model so the data carries a learnable signal.
2. **Exploratory data analysis** — interactive Plotly charts of readmission
   rates by admission type and department, plus a correlation heatmap.
3. **Predictive modeling** — a balanced `RandomForestClassifier` inside a
   scikit-learn pipeline with one-hot encoding for categoricals.
4. **Feature importance** — permutation importance on held-out data, reported
   per real-world factor for clinical interpretability, plus a ROC curve.
5. **Interactive What-If dashboard** — sidebar sliders for age, length of stay,
   comorbidities, admission type, and department that update a live risk gauge.

## 1. Install the required libraries

```bash
pip install -r requirements.txt
```

Or install them individually:

```bash
pip install pandas scikit-learn streamlit plotly
```

> Tip: use a virtual environment to keep dependencies isolated:
> ```bash
> python -m venv .venv
> # Windows (PowerShell)
> .venv\Scripts\Activate.ps1
> # macOS / Linux
> source .venv/bin/activate
> ```

## 2. Run the app

From the project directory, run:

```bash
streamlit run app.py
```

Streamlit will start a local server and open the app in your browser
(usually at http://localhost:8501).

## Notes

- The dataset is **fully synthetic** and intended for demonstration only —
  it is not derived from real patients and must not be used clinically.
- Data generation and model training are cached, so the app loads instantly
  after the first run and the What-If calculator responds in real time.
