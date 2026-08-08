import io
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.data.preprocessing import load_preprocessor, get_transformed_feature_names
from src.explainability.explainability import explain_patient_risk, get_global_feature_importance
from src.monitoring.drift_monitor import analyze_dataset_drift

# Initialize FastAPI App
app = FastAPI(
    title="MedPulse AI — Enterprise Medical Predictive Analytics API",
    description="Production REST API for Clinical Cardiac Risk Inference, Explainable AI (XAI), and MLOps Drift Monitoring.",
    version="1.0.0"
)

# Enable CORS for local web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Cached Objects
preprocessor = None
champion_model = None
feature_names = []
baseline_df = None

def get_loaded_artifacts():
    global preprocessor, champion_model, feature_names, baseline_df
    if preprocessor is None:
        try:
            preprocessor = load_preprocessor()
            champion_model = joblib.load(config.MODELS_DIR / "champion_model.joblib")
            feature_names = get_transformed_feature_names(preprocessor)
            if config.RAW_DATA_PATH.exists():
                baseline_df = pd.read_csv(config.RAW_DATA_PATH)
        except Exception as e:
            print(f"[!] Warning loading model artifacts: {e}")

@app.on_event("startup")
def startup_event():
    get_loaded_artifacts()

class PatientDataInput(BaseModel):
    age: int = Field(54, ge=18, le=100)
    sex: str = Field("Male",pattern="^(Male|Female)$")
    chest_pain_type: str = Field("Typical Angina")
    resting_bp: float = Field(138.0, ge=70, le=240)
    cholesterol: float = Field(245.0, ge=100, le=600)
    fasting_bs: float = Field(115.0, ge=50, le=400)
    resting_ecg: str = Field("Normal")
    max_hr: float = Field(142.0, ge=60, le=220)
    exercise_angina: str = Field("Yes", pattern="^(Yes|No)$")
    oldpeak: float = Field(1.8, ge=0.0, le=10.0)
    st_slope: str = Field("Flat")
    bmi: float = Field(29.4, ge=12.0, le=60.0)
    hba1c: float = Field(6.4, ge=3.0, le=15.0)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "MedPulse AI Analytics Engine",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.post("/api/predict")
def predict_single_patient(patient: PatientDataInput):
    get_loaded_artifacts()
    if champion_model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Models not loaded. Train pipeline first.")

    patient_dict = patient.dict()
    df_single = pd.DataFrame([patient_dict])

    explanation = explain_patient_risk(df_single, preprocessor, champion_model, feature_names)
    
    return {
        "status": "success",
        "patient_input": patient_dict,
        "assessment": explanation
    }

@app.post("/api/predict-batch")
async def predict_batch_csv(file: UploadFile = File(...)):
    get_loaded_artifacts()
    if champion_model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Models not loaded. Train pipeline first.")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV format.")

    content = await file.read()
    df_batch = pd.read_csv(io.StringIO(content.decode('utf-8')))

    # Verify required feature columns exist
    missing_cols = [c for c in config.FEATURE_COLUMNS if c not in df_batch.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing required columns in CSV: {missing_cols}")

    transformed = preprocessor.transform(df_batch[config.FEATURE_COLUMNS])
    probas = champion_model.predict_proba(transformed)[:, 1]

    results = []
    low_cnt, mod_cnt, high_cnt = 0, 0, 0

    for idx, prob in enumerate(probas):
        prob_val = float(prob)
        if prob_val < config.RISK_LOW_MAX:
            cat = "Low Risk"
            low_cnt += 1
        elif prob_val < config.RISK_MODERATE_MAX:
            cat = "Moderate Risk"
            mod_cnt += 1
        else:
            cat = "High Risk"
            high_cnt += 1

        results.append({
            "patient_id": idx + 1,
            "age": int(df_batch["age"].iloc[idx]) if "age" in df_batch.columns else 0,
            "sex": str(df_batch["sex"].iloc[idx]) if "sex" in df_batch.columns else "N/A",
            "risk_probability": round(prob_val, 4),
            "risk_percentage": round(prob_val * 100, 1),
            "risk_category": cat
        })

    return {
        "status": "success",
        "total_records": len(results),
        "summary": {
            "low_risk_count": low_cnt,
            "moderate_risk_count": mod_cnt,
            "high_risk_count": high_cnt
        },
        "predictions": results
    }

@app.get("/api/metrics")
def get_model_metrics():
    report_path = config.ARTIFACTS_DIR / "evaluation_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation report not found. Train pipeline first.")
    
    with open(report_path, "r") as f:
        data = json.load(f)
    return data

@app.get("/api/explain/global")
def get_global_explanations():
    get_loaded_artifacts()
    if champion_model is None:
        raise HTTPException(status_code=500, detail="Champion model not loaded.")
    
    importance = get_global_feature_importance(champion_model, feature_names)
    return {
        "status": "success",
        "global_feature_importance": importance
    }

@app.post("/api/drift")
def check_data_drift(sample_size: int = 200, drift_multiplier: float = 1.0):
    get_loaded_artifacts()
    if baseline_df is None:
        raise HTTPException(status_code=500, detail="Baseline training dataset unavailable.")

    # Create sample current batch with optional drift injection
    current_sample = baseline_df.sample(n=min(sample_size, len(baseline_df)), random_state=123).copy()

    if drift_multiplier != 1.0:
        # Simulate shift in key features
        current_sample["age"] = (current_sample["age"] * drift_multiplier).clip(18, 90)
        current_sample["resting_bp"] = (current_sample["resting_bp"] * drift_multiplier).clip(90, 220)
        current_sample["hba1c"] = (current_sample["hba1c"] * drift_multiplier).clip(4.0, 14.0)

    drift_report = analyze_dataset_drift(baseline_df, current_sample)
    return drift_report
