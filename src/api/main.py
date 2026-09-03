import io
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# =========================================================
# PROJECT IMPORTS
# =========================================================

from src import config

from src.data.preprocessing import (
    load_preprocessor,
    get_transformed_feature_names
)

from src.explainability.explainability import (
    explain_patient_risk,
    get_global_feature_importance
)

from src.monitoring.drift_monitor import (
    analyze_dataset_drift
)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="MedPulse AI — Enterprise Medical Predictive Analytics API",
    description=(
        "Production REST API for Clinical Cardiac Risk "
        "Inference, Explainable AI (XAI), and MLOps "
        "Drift Monitoring."
    ),
    version="1.0.0"
)


# =========================================================
# CORS CONFIGURATION
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "https://medpulse-ai-frontend.onrender.com"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =========================================================
# GLOBAL CACHED OBJECTS
# =========================================================

preprocessor = None
champion_model = None
feature_names = []
baseline_df = None

# SHAP representative background/reference data
background_data = None


# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================

def get_loaded_artifacts():
    """
    Load all required MedPulse AI artifacts.

    Loads:
        1. Preprocessor
        2. Champion model
        3. Transformed feature names
        4. Baseline/raw dataset
        5. SHAP background data
    """

    global preprocessor
    global champion_model
    global feature_names
    global baseline_df
    global background_data

    # -----------------------------------------------------
    # Avoid loading repeatedly
    # -----------------------------------------------------

    if (
        preprocessor is not None
        and champion_model is not None
        and background_data is not None
    ):
        return

    try:

        print("[INFO] Loading MedPulse AI artifacts...")

        # =================================================
        # 1. LOAD PREPROCESSOR
        # =================================================

        preprocessor = load_preprocessor()

        if preprocessor is None:
            raise RuntimeError(
                "Preprocessor could not be loaded."
            )

        print(
            "[INFO] Preprocessor loaded successfully."
        )

        # =================================================
        # 2. LOAD CHAMPION MODEL
        # =================================================

        model_path = (
            config.MODELS_DIR
            / "champion_model.joblib"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                "Champion model not found at: "
                f"{model_path}"
            )

        champion_model = joblib.load(model_path)

        if champion_model is None:
            raise RuntimeError(
                "Champion model was loaded as None."
            )

        print(
            "[INFO] Champion model loaded successfully."
        )

        # =================================================
        # 3. GET FEATURE NAMES
        # =================================================

        feature_names = get_transformed_feature_names(
            preprocessor
        )

        if (
            feature_names is None
            or len(feature_names) == 0
        ):
            raise RuntimeError(
                "Could not obtain transformed feature names."
            )

        print(
            "[INFO] Transformed features loaded: "
            f"{len(feature_names)}"
        )

        # =================================================
        # 4. LOAD RAW DATASET
        # =================================================

        if not config.RAW_DATA_PATH.exists():
            raise FileNotFoundError(
                "Raw medical dataset not found at: "
                f"{config.RAW_DATA_PATH}"
            )

        baseline_df = pd.read_csv(
            config.RAW_DATA_PATH
        )

        if baseline_df.empty:
            raise RuntimeError(
                "The raw medical dataset is empty."
            )

        print(
            "[INFO] Baseline dataset loaded: "
            f"{len(baseline_df)} rows, "
            f"{len(baseline_df.columns)} columns."
        )

        # =================================================
        # 5. CHECK FEATURE COLUMNS
        # =================================================

        missing_columns = [
            column
            for column in config.FEATURE_COLUMNS
            if column not in baseline_df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Required feature columns are missing "
                f"from the dataset: {missing_columns}"
            )

        # =================================================
        # 6. CREATE SHAP BACKGROUND DATA
        # =================================================

        background_raw = (
            baseline_df[
                config.FEATURE_COLUMNS
            ]
            .sample(
                n=min(
                    100,
                    len(baseline_df)
                ),
                random_state=42
            )
            .copy()
        )

        if background_raw.empty:
            raise RuntimeError(
                "SHAP background dataset could not be created."
            )

        print(
            "[INFO] SHAP background samples: "
            f"{len(background_raw)}"
        )

        # =================================================
        # 7. APPLY SAME PREPROCESSING
        # =================================================

        background_data = preprocessor.transform(
            background_raw
        )

        if background_data is None:
            raise RuntimeError(
                "SHAP background data transformation failed."
            )

        print(
            "[INFO] SHAP background data prepared successfully."
        )

        # =================================================
        # 8. FINAL VALIDATION
        # =================================================

        if preprocessor is None:
            raise RuntimeError(
                "Preprocessor validation failed."
            )

        if champion_model is None:
            raise RuntimeError(
                "Champion model validation failed."
            )

        if background_data is None:
            raise RuntimeError(
                "SHAP background data validation failed."
            )

        print(
            "[INFO] All MedPulse AI artifacts loaded successfully."
        )

    except FileNotFoundError as e:

        preprocessor = None
        champion_model = None
        feature_names = []
        baseline_df = None
        background_data = None

        print(
            f"[ERROR] File not found: {e}"
        )

        raise RuntimeError(
            f"Required MedPulse AI file is missing: {e}"
        ) from e

    except ValueError as e:

        preprocessor = None
        champion_model = None
        feature_names = []
        baseline_df = None
        background_data = None

        print(
            f"[ERROR] Data/configuration error: {e}"
        )

        raise RuntimeError(
            f"MedPulse AI data configuration error: {e}"
        ) from e

    except Exception as e:

        preprocessor = None
        champion_model = None
        feature_names = []
        baseline_df = None
        background_data = None

        print(
            "[ERROR] Failed to load MedPulse AI artifacts: "
            f"{e}"
        )

        raise RuntimeError(
            f"Failed to initialize MedPulse AI: {e}"
        ) from e


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup_event():
    print("[INFO] MedPulse AI API started successfully.")
    print("[INFO] Model artifacts will be loaded when required.")

# =========================================================
# PATIENT INPUT MODEL
# =========================================================

class PatientDataInput(BaseModel):

    age: int = Field(
        54,
        ge=18,
        le=100
    )

    sex: str = Field(
        "Male",
        pattern="^(Male|Female)$"
    )

    chest_pain_type: str = Field(
        "Typical Angina"
    )

    resting_bp: float = Field(
        138.0,
        ge=70,
        le=240
    )

    cholesterol: float = Field(
        245.0,
        ge=100,
        le=600
    )

    fasting_bs: float = Field(
        115.0,
        ge=50,
        le=400
    )

    resting_ecg: str = Field(
        "Normal"
    )

    max_hr: float = Field(
        142.0,
        ge=60,
        le=220
    )

    exercise_angina: str = Field(
        "Yes",
        pattern="^(Yes|No)$"
    )

    oldpeak: float = Field(
        1.8,
        ge=0.0,
        le=10.0
    )

    st_slope: str = Field(
        "Flat"
    )

    bmi: float = Field(
        29.4,
        ge=12.0,
        le=60.0
    )

    hba1c: float = Field(
        6.4,
        ge=3.0,
        le=15.0
    )


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def read_root():

    return {
        "status": "online",
        "service": "MedPulse AI Analytics Engine",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    try:

        get_loaded_artifacts()

        return {
            "status": "healthy",
            "model_loaded": champion_model is not None,
            "preprocessor_loaded": preprocessor is not None,
            "shap_background_loaded": background_data is not None
        }

    except Exception as e:

        return {
            "status": "unhealthy",
            "model_loaded": False,
            "preprocessor_loaded": False,
            "shap_background_loaded": False,
            "error": str(e)
        }


# =========================================================
# SINGLE PATIENT PREDICTION
# =========================================================

@app.post("/api/predict")
def predict_single_patient(
    patient: PatientDataInput
):

    try:

        get_loaded_artifacts()

        if (
            champion_model is None
            or preprocessor is None
        ):
            raise HTTPException(
                status_code=500,
                detail=(
                    "Models not loaded. "
                    "Train pipeline first."
                )
            )

        if background_data is None:

            raise HTTPException(
                status_code=500,
                detail=(
                    "SHAP background data is not loaded."
                )
            )

        # -------------------------------------------------
        # Convert patient to dictionary
        # -------------------------------------------------

        try:

            patient_dict = patient.model_dump()

        except AttributeError:

            patient_dict = patient.dict()

        # -------------------------------------------------
        # DataFrame
        # -------------------------------------------------

        df_single = pd.DataFrame(
            [patient_dict]
        )

        # -------------------------------------------------
        # Required columns
        # -------------------------------------------------

        missing_columns = [
            column
            for column in config.FEATURE_COLUMNS
            if column not in df_single.columns
        ]

        if missing_columns:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing required patient fields: "
                    f"{missing_columns}"
                )
            )

        # -------------------------------------------------
        # SHAP explanation
        # -------------------------------------------------

        explanation = explain_patient_risk(
            df_single,
            preprocessor,
            champion_model,
            feature_names,
            background_data
        )

        return {
            "status": "success",
            "patient_input": patient_dict,
            "assessment": explanation
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "[ERROR] Single patient prediction failed: "
            f"{e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction/explanation failed: "
                f"{str(e)}"
            )
        )


# =========================================================
# BATCH CSV PREDICTION
# =========================================================

@app.post("/api/predict-batch")
async def predict_batch_csv(
    file: UploadFile = File(...)
):

    try:

        get_loaded_artifacts()

        if (
            champion_model is None
            or preprocessor is None
        ):

            raise HTTPException(
                status_code=500,
                detail="Models not loaded."
            )

        # -------------------------------------------------
        # Validate filename
        # -------------------------------------------------

        if not file.filename:

            raise HTTPException(
                status_code=400,
                detail="No file was provided."
            )

        if not file.filename.lower().endswith(".csv"):

            raise HTTPException(
                status_code=400,
                detail="File must be a CSV format."
            )

        # -------------------------------------------------
        # Read file
        # -------------------------------------------------

        content = await file.read()

        if not content:

            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV file is empty."
            )

        try:

            df_batch = pd.read_csv(
                io.StringIO(
                    content.decode("utf-8")
                )
            )

        except UnicodeDecodeError:

            raise HTTPException(
                status_code=400,
                detail="CSV file must use UTF-8 encoding."
            )

        except Exception as e:

            raise HTTPException(
                status_code=400,
                detail=f"Could not read CSV file: {e}"
            )

        # -------------------------------------------------
        # Empty dataframe
        # -------------------------------------------------

        if df_batch.empty:

            raise HTTPException(
                status_code=400,
                detail="CSV file contains no records."
            )

        # -------------------------------------------------
        # Required columns
        # -------------------------------------------------

        missing_cols = [
            column
            for column in config.FEATURE_COLUMNS
            if column not in df_batch.columns
        ]

        if missing_cols:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing required columns in CSV: "
                    f"{missing_cols}"
                )
            )

        # -------------------------------------------------
        # Select features
        # -------------------------------------------------

        batch_features = df_batch[
            config.FEATURE_COLUMNS
        ]

        # -------------------------------------------------
        # Preprocess
        # -------------------------------------------------

        transformed = preprocessor.transform(
            batch_features
        )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        if not hasattr(
            champion_model,
            "predict_proba"
        ):

            raise HTTPException(
                status_code=500,
                detail=(
                    "Champion model does not support "
                    "probability prediction."
                )
            )

        probas = champion_model.predict_proba(
            transformed
        )[:, 1]

        # -------------------------------------------------
        # Prepare results
        # -------------------------------------------------

        results = []

        low_cnt = 0
        mod_cnt = 0
        high_cnt = 0

        for index, probability in enumerate(probas):

            probability = float(probability)

            percentage = probability * 100.0

            if probability < 0.30:

                risk_level = "Low"
                low_cnt += 1

            elif probability < 0.70:

                risk_level = "Moderate"
                mod_cnt += 1

            else:

                risk_level = "High"
                high_cnt += 1

            results.append(
                {
                    "row": int(index + 1),
                    "risk_probability": round(
                        probability,
                        4
                    ),
                    "risk_percentage": round(
                        percentage,
                        2
                    ),
                    "risk_level": risk_level
                }
            )

        return {
            "status": "success",
            "total_records": len(results),
            "summary": {
                "low_risk": low_cnt,
                "moderate_risk": mod_cnt,
                "high_risk": high_cnt
            },
            "results": results
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "[ERROR] Batch prediction failed: "
            f"{e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Batch prediction failed: "
                f"{str(e)}"
            )
        )


# =========================================================
# GLOBAL FEATURE IMPORTANCE
# =========================================================

@app.get("/api/explain/global")
def global_explanation():

    try:

        get_loaded_artifacts()

        if (
            champion_model is None
            or preprocessor is None
        ):

            raise HTTPException(
                status_code=500,
                detail="Models not loaded."
            )

        if background_data is None:

            raise HTTPException(
                status_code=500,
                detail=(
                    "SHAP background data is not loaded."
                )
            )

        # -------------------------------------------------
        # Calculate global importance
        # -------------------------------------------------

        importance = get_global_feature_importance(
            champion_model,
            preprocessor,
            feature_names,
            background_data
        )

        return {
            "status": "success",
            "global_feature_importance": importance
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "[ERROR] Global explanation failed: "
            f"{e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Global explanation failed: "
                f"{str(e)}"
            )
        )


# =========================================================
# DRIFT MONITORING
# =========================================================

@app.post("/api/drift")
async def dataset_drift(
    file: UploadFile = File(...)
):

    try:

        get_loaded_artifacts()

        if baseline_df is None:

            raise HTTPException(
                status_code=500,
                detail="Baseline dataset is not loaded."
            )

        if not file.filename:

            raise HTTPException(
                status_code=400,
                detail="No file was provided."
            )

        if not file.filename.lower().endswith(".csv"):

            raise HTTPException(
                status_code=400,
                detail="File must be a CSV format."
            )

        content = await file.read()

        if not content:

            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV file is empty."
            )

        try:

            current_df = pd.read_csv(
                io.StringIO(
                    content.decode("utf-8")
                )
            )

        except Exception as e:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not read CSV file: {e}"
                )
            )

        if current_df.empty:

            raise HTTPException(
                status_code=400,
                detail="Uploaded dataset is empty."
            )

        # -------------------------------------------------
        # Run existing drift monitor
        # -------------------------------------------------

        try:

            drift_result = analyze_dataset_drift(
                baseline_df,
                current_df
            )

        except TypeError:

            # Compatibility with implementations that
            # expect only feature columns.
            drift_result = analyze_dataset_drift(
                baseline_df[
                    config.FEATURE_COLUMNS
                ],
                current_df[
                    config.FEATURE_COLUMNS
                ]
            )

        return {
            "status": "success",
            "drift_analysis": drift_result
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "[ERROR] Drift analysis failed: "
            f"{e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Drift analysis failed: "
                f"{str(e)}"
            )
        )


# =========================================================
# API INFORMATION
# =========================================================

@app.get("/api/info")
def api_info():

    return {
        "project": "MedPulse AI",
        "version": "1.0.0",
        "description": (
            "Clinical Cardiac Risk Prediction, "
            "Explainable AI and MLOps Drift Monitoring"
        ),
        "endpoints": [
            "/",
            "/health",
            "/api/info",
            "/api/predict",
            "/api/predict-batch",
            "/api/explain/global",
            "/api/drift",
            "/docs"
        ]
    }


# =========================================================
# LOCAL DEVELOPMENT
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )