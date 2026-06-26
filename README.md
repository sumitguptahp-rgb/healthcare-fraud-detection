# 🏥 Healthcare Fraud Detection

A machine learning system for detecting fraudulent healthcare insurance claims, built with XGBoost and explained via SHAP. Features a FastAPI backend and an interactive web-based frontend.

---

## 🔍 Overview

This project trains a fraud detection classifier on anonymized healthcare claim data and exposes it via a REST API. The frontend allows investigators to submit a claim and instantly receive a fraud risk score, band classification, and the top contributing factors (powered by SHAP).

### Key Features
- **XGBoost classifier** trained on engineered features (velocity anomalies, approval fractions, provider surge indicators, etc.)
- **SHAP explainability** — every prediction comes with human-readable reasoning
- **FastAPI backend** with `/predict` endpoint
- **Interactive frontend** served directly from the backend

---

## 🗂️ Project Structure

```
├── backend/
│   └── app.py              # FastAPI application
├── frontend/               # HTML/CSS/JS UI
├── models/                 # Trained model artifacts (not tracked in git)
│   ├── xgboost_model.joblib
│   ├── scaler.joblib
│   ├── imputer.joblib
│   ├── columns.json
│   ├── mappings.json
│   └── thresholds.json
├── fraud-detection-healthcare_obfuscated.ipynb  # Training notebook
├── requirements.txt
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/healthcare-fraud-detection.git
cd healthcare-fraud-detection
```

### 2. Create a virtual environment & install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# or: source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 4. Start the backend
```bash
cd backend
python app.py
```

The app will be available at **http://localhost:8000**

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost, scikit-learn, SHAP |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | HTML, CSS, Vanilla JS |
| Data | pandas, numpy, imbalanced-learn |

---

## 📄 License

MIT
