# 🫁 NAFLD Detection System — Streamlit Dashboard

ML-powered clinical decision support tool for Non-Alcoholic Fatty Liver Disease (NAFLD) risk prediction.

---

## 📌 Project Information

This project was developed as part of a group academic project for the Applied Machine Learning course.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📁 Project Structure

```
your_project/
├── app.py                      ← Streamlit dashboard (this file)
├── best_model_NAFLD.joblib     ← Your saved ML model
├── scaler_NAFLD.joblib         ← Your saved StandardScaler
├── requirements.txt
└── README.md
```

> **Important:** Place `best_model_NAFLD.joblib` and `scaler_NAFLD.joblib`
> in the **same folder** as `app.py` before running.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** automatically.

---

## ⚙️ Feature Alignment

The constant `TRAINING_FEATURES` (top of `app.py`) must match the **exact column order** your model was trained on.

```python
TRAINING_FEATURES = [
    "Age", "Sex", "BMI", "Waist_Circumference",
    "Fasting_Blood_Sugar", "Triglycerides", "Total_Cholesterol",
    "Insulin", "CRP", "CAP_Score", "Fibroscan_kPa",
    "ALT", "AST", "GGT",
]
```

If your training dataset used different column names or a different order, update this list and the `FEATURE_DEFAULTS` dictionary accordingly.

---

## 🐛 Bug Fix Applied

The merged line from the original ML notebook:

```python
# ❌ Buggy (merged):
X_test_scaled = scaler.transform(X_test)corr = df.corr()
```

has been correctly separated in this dashboard:

```python
# ✅ Fixed:
X_test_scaled = scaler.transform(X_test)   # scale test features
# corr = df.corr()                         # EDA step — not needed at inference
```

---

## 🎯 Project Features

- Machine Learning based NAFLD risk prediction
- Interactive Streamlit dashboard
- User-friendly clinical input interface
- Data preprocessing and feature scaling
- Model loading using Joblib
- Real-time prediction generation

---

## ⚕️ Disclaimer

This tool is for **educational and academic purposes only**.

It is not a certified medical device and must not replace clinical judgment.
