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

# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------
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

    # Frontend development URLs
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
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

# IMPORTANT:
# SHAP requires representative background/reference data.
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

    The SHAP background data is created from a representative
    sample of the raw dataset and transformed using the SAME
    preprocessing pipeline used during model training.
    """

    global preprocessor
    global champion_model
    global feature_names
    global baseline_df
    global background_data

    # -----------------------------------------------------
    # Avoid loading everything repeatedly
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

        champion_model = joblib.load(
            model_path
        )

        if champion_model is None:

            raise RuntimeError(
                "Champion model was loaded as None."
            )

        print(
            "[INFO] Champion model loaded successfully."
        )


        # =================================================
        # 3. GET TRANSFORMED FEATURE NAMES
        # =================================================

        feature_names = (
            get_transformed_feature_names(
                preprocessor
            )
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
        # 4. LOAD RAW / BASELINE DATASET
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
        # 5. CHECK REQUIRED FEATURE COLUMNS
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

        # Select at most 100 representative samples.
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

        # IMPORTANT:
        # The background data must be transformed using
        # the SAME preprocessor used during model training.

        background_data = (
            preprocessor.transform(
                background_raw
            )
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


    # =====================================================
    # FILE ERROR
    # =====================================================

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


    # =====================================================
    # DATA / CONFIGURATION ERROR
    # =====================================================

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


    # =====================================================
    # GENERAL ERROR
    # =====================================================

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
# STARTUP EVENT
# =========================================================

@app.on_event("startup")
def startup_event():

    try:

        get_loaded_artifacts()

        print(
            "[INFO] MedPulse AI startup completed successfully."
        )

    except Exception as e:

        print(
            "[ERROR] Startup artifact loading failed: "
            f"{e}"
        )

        # We don't hide the error.
        # API endpoints will provide an appropriate error
        # if artifacts are unavailable.


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
# SINGLE PATIENT PREDICTION
# =========================================================

@app.post("/api/predict")
def predict_single_patient(
    patient: PatientDataInput
):

    try:

        # Load artifacts
        get_loaded_artifacts()

        # -------------------------------------------------
        # Validate model
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Validate SHAP background data
        # -------------------------------------------------

        if background_data is None:

            raise HTTPException(
                status_code=500,
                detail=(
                    "SHAP background data is not loaded."
                )
            )

        # -------------------------------------------------
        # Convert patient input to dictionary
        # -------------------------------------------------

        try:

            # Pydantic v2
            patient_dict = patient.model_dump()

        except AttributeError:

            # Pydantic v1 compatibility
            patient_dict = patient.dict()

        # -------------------------------------------------
        # Convert to DataFrame
        # -------------------------------------------------

        df_single = pd.DataFrame(
            [patient_dict]
        )

        # -------------------------------------------------
        # Verify required columns
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
        # Generate SHAP explanation
        # -------------------------------------------------

        explanation = explain_patient_risk(
            df_single,
            preprocessor,
            champion_model,
            feature_names,
            background_data
        )

        # -------------------------------------------------
        # Return result
        # -------------------------------------------------

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

        # Load artifacts
        get_loaded_artifacts()

        # -------------------------------------------------
        # Validate model
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Validate file name
        # -------------------------------------------------

        if not file.filename:

            raise HTTPException(
                status_code=400,
                detail="No file was provided."
            )

        if not file.filename.lower().endswith(
            ".csv"
        ):

            raise HTTPException(
                status_code=400,
                detail="File must be a CSV format."
            )

        # -------------------------------------------------
        # Read uploaded file
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
                detail=(
                    "CSV file must use UTF-8 encoding."
                )
            )

        except Exception as e:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not read CSV file: {e}"
                )
            )

        # -------------------------------------------------
        # Check empty dataframe
        # -------------------------------------------------

        if df_batch.empty:

            raise HTTPException(
                status_code=400,
                detail="CSV file contains no records."
            )

        # -------------------------------------------------
        # Verify required feature columns
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
        # Select feature columns
        # -------------------------------------------------

        batch_features = (
            df_batch[
                config.FEATURE_COLUMNS
            ]
        )

        # -------------------------------------------------
        # Apply preprocessing
        # -------------------------------------------------

        transformed = (
            preprocessor.transform(
                batch_features
            )
        )

        # -------------------------------------------------
        # Predict probabilities
        # -------------------------------------------------

        probas = (
            champion_model.predict_proba(
                transformed
            )[:, 1]
        )

        # -------------------------------------------------
        # Prepare results
        # -------------------------------------------------

        results = []

        low_cnt = 0
        mod_cnt = 0
        high_cnt = 0

        for idx, prob in enumerate(
            probas
        ):

            prob_val = float(prob)

            # ---------------------------------------------
            # Risk classification
            # ---------------------------------------------

            if (
                prob_val
                < config.RISK_LOW_MAX
            ):

                category = "Low Risk"
                low_cnt += 1

            elif (
                prob_val
                < config.RISK_MODERATE_MAX
            ):

                category = "Moderate Risk"
                mod_cnt += 1

            else:

                category = "High Risk"
                high_cnt += 1

            # ----------