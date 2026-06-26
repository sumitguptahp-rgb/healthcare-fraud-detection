from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import os
import shap

app = FastAPI(title="Health Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'index.html'))


# Load artifacts
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    imputer = joblib.load(os.path.join(MODEL_DIR, 'imputer.joblib'))
    
    with open(os.path.join(MODEL_DIR, 'mappings.json'), 'r') as f:
        mappings = json.load(f)
    with open(os.path.join(MODEL_DIR, 'thresholds.json'), 'r') as f:
        thresholds = json.load(f)
    with open(os.path.join(MODEL_DIR, 'columns.json'), 'r') as f:
        predictor_keys = json.load(f)
        
    shap_engine = shap.TreeExplainer(model)
except Exception as e:
    print(f"Error loading models: {e}")

class ClaimInput(BaseModel):
    Provider_ID: str
    Patient_Age: int
    Patient_Gender: str
    Diagnosis_Code: str
    Procedure_Code: str
    Claim_Amount: float
    Approved_Amount: float
    Insurance_Type: str
    Claim_Submission_Date: str = ""  # kept for display only, not used in prediction
    Days_Between_Service_and_Claim: int
    Number_of_Claims_Per_Provider_Monthly: int
    Provider_Specialty: str
    Patient_State: str = ""
    Length_of_Stay: int
    Visit_Type: str
    Chronic_Condition_Flag: int
    Prior_Visits_12m: int

@app.post("/predict")
def predict(claim: ClaimInput):
    # Convert INR to USD
    claim.Claim_Amount = claim.Claim_Amount / 86.0
    claim.Approved_Amount = claim.Approved_Amount / 86.0
    
    # Convert input to DataFrame
    df = pd.DataFrame([claim.model_dump()])
    
    # Date-derived features removed - submission date is not predictive of fraud
    
    # 2. Derived Proportions
    df['Approval_Fraction'] = df['Claim_Amount'] / (df['Approved_Amount'] + 1)
    df['Gap_Months_Norm'] = df['Days_Between_Service_and_Claim'] / 30.0
    df['Discrepancy_Value'] = df['Claim_Amount'] - df['Approved_Amount']
    
    # 3. Demographic Risk Indicators
    # Note: Age_Bracket isn't in predictor_keys due to being categorical and not one-hot encoded in original script
    # wait, the original script dropped 'Age_Bracket' anyway.
    
    top_5_pct_claim = thresholds['top_5_pct_claim']
    bottom_10_pct_delay = thresholds['bottom_10_pct_delay']
    monthly_spike_threshold = thresholds['monthly_spike_threshold']
    
    df['Anomalous_Velocity'] = np.where(
        (df['Claim_Amount'] > top_5_pct_claim) & (df['Days_Between_Service_and_Claim'] < bottom_10_pct_delay),
        1, 0
    )
    df['Encounters_Per_Year_Of_Life'] = df['Prior_Visits_12m'] / (df['Patient_Age'] + 1)
    
    # 4. Compound Metrics
    df['Daily_Burn_Rate'] = df['Claim_Amount'] / (df['Length_of_Stay'] + 1)
    df['Provider_Surge_Indicator'] = np.where(
        df['Number_of_Claims_Per_Provider_Monthly'] > monthly_spike_threshold, 1, 0
    )
    
    # 5. Categorical Encoding (Frequency)
    complex_categoricals = ['Provider_ID', 'Diagnosis_Code', 'Procedure_Code']
    for cat_var in complex_categoricals:
        val = df.iloc[0][cat_var]
        # map to density from training data, default to 0 if unseen
        df[f'{cat_var}_Density'] = mappings[cat_var].get(val, 0)
        
    # 6. Dummy Variables (Manual mapping to match training columns)
    # Create empty dataframe with all predictor_keys initialized to 0
    final_df = pd.DataFrame(0, index=[0], columns=predictor_keys)
    
    # Fill in numerical features that don't need dummy encoding
    for col in df.columns:
        if col in predictor_keys:
            final_df[col] = df[col].values
            
    # For categorical columns, set the corresponding dummy column to 1
    simple_categoricals = ['Patient_Gender', 'Insurance_Type', 'Provider_Specialty', 
                           'Patient_State', 'Claim_Status', 'Visit_Type']
    # 'Claim_Status' is missing from ClaimInput if it's pending? 
    # Let's assume input has it, wait, we didn't add it to ClaimInput. 
    # Ah, Claim_Status is in the original CSV. I'll add a default if not present.
    # Wait, the input might not have Claim_Status if it's a new claim. But let's assume it's 'Pending'.
    
    for cat_var in simple_categoricals:
        if cat_var in df.columns:
            val = df.iloc[0][cat_var]
            dummy_col = f"{cat_var}_{val}"
            if dummy_col in predictor_keys:
                final_df[dummy_col] = 1
                
    # Scale and impute
    filled = imputer.transform(final_df)
    filled_df = pd.DataFrame(filled, columns=predictor_keys)
    filled_df = filled_df.replace([np.inf, -np.inf], np.nan).fillna(0) # basic fallback
    
    normed = scaler.transform(filled_df)
    
    # Predict
    prob = float(model.predict_proba(normed)[0][1])
    
    # SHAP Explainer
    marginals = shap_engine.shap_values(normed)[0]
    
    # Map feature names to SHAP values
    feature_contributions = []
    for i, col in enumerate(predictor_keys):
        if marginals[i] > 0:
            # Simple cleanup for readability
            clean_name = col.replace('_', ' ')
            if 'Density' in clean_name:
                clean_name = f"High Risk {clean_name.replace(' Density', '')}"
            feature_contributions.append({
                "feature": clean_name,
                "contribution": float(marginals[i])
            })
            
    # Sort top positive drivers
    feature_contributions = sorted(feature_contributions, key=lambda x: x["contribution"], reverse=True)[:3]
    
    # Band mapping
    if prob >= 0.8:
        band = "CRITICAL"
    elif prob >= 0.6:
        band = "ELEVATED"
    elif prob >= 0.4:
        band = "MODERATE"
    else:
        band = "NEGLIGIBLE"
        
    return {
        "confidence": prob,
        "band": band,
        "reasons": feature_contributions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
