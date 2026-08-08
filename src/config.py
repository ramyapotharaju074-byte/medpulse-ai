import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models_store"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Create directories if they do not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Specs
RAW_DATA_PATH = DATA_DIR / "raw_medical_dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_medical_dataset.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

# Target & Feature Specs
TARGET_COL = "cardiac_risk"

NUMERICAL_FEATURES = [
    "age",
    "resting_bp",
    "cholesterol",
    "fasting_bs",
    "max_hr",
    "oldpeak",
    "bmi",
    "hba1c"
]

CATEGORICAL_FEATURES = [
    "sex",
    "chest_pain_type",
    "resting_ecg",
    "exercise_angina",
    "st_slope"
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Risk Classification Thresholds
RISK_LOW_MAX = 0.35
RISK_MODERATE_MAX = 0.70

# Feature Display Labels for UI & XAI
FEATURE_DISPLAY_NAMES = {
    "age": "Age (years)",
    "resting_bp": "Resting BP (mm Hg)",
    "cholesterol": "Serum Cholesterol (mg/dL)",
    "fasting_bs": "Fasting Blood Sugar (mg/dL)",
    "max_hr": "Maximum Heart Rate Achieved",
    "oldpeak": "ST Depression (Oldpeak)",
    "bmi": "Body Mass Index (BMI)",
    "hba1c": "HbA1c Level (%)",
    "sex": "Biological Sex",
    "chest_pain_type": "Chest Pain Severity Type",
    "resting_ecg": "Resting ECG Results",
    "exercise_angina": "Exercise-Induced Angina",
    "st_slope": "Peak Exercise ST Segment Slope"
}
