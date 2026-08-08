import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure src module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def generate_synthetic_medical_data(num_samples: int = 2500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic, multi-variable medical dataset for cardiac risk assessment.
    Incorporate non-linear physiological relationships, correlations, and target risk probabilities.
    """
    np.random.seed(random_seed)

    # Demographic & Biomarkers
    age = np.random.normal(54, 11, num_samples).clip(28, 85).astype(int)
    sex = np.random.choice(["Male", "Female"], size=num_samples, p=[0.58, 0.42])
    
    # Blood Pressure (Higher for older age)
    bp_base = 110 + (age * 0.4) + np.random.normal(0, 12, num_samples)
    resting_bp = bp_base.clip(90, 200).round(1)

    # Cholesterol (mg/dL)
    chol_base = 180 + (age * 0.5) + np.random.normal(0, 35, num_samples)
    cholesterol = chol_base.clip(120, 450).round(1)

    # Fasting Blood Sugar (mg/dL)
    fasting_bs = np.random.gamma(shape=9.0, scale=12.0, size=num_samples).clip(70, 280).round(1)

    # BMI
    bmi = np.random.normal(27.5, 4.8, num_samples).clip(16.0, 48.0).round(1)

    # HbA1c Level (%)
    hba1c = np.random.normal(5.8, 1.1, num_samples).clip(4.2, 12.5).round(2)

    # Max Heart Rate (Decreases with age)
    max_hr = (210 - (age * 0.7) + np.random.normal(0, 15, num_samples)).clip(75, 205).round(1)

    # ST Depression (oldpeak)
    oldpeak = np.random.exponential(scale=0.8, size=num_samples).clip(0.0, 6.2).round(2)

    # Categorical Clinical Variables
    chest_pain_type = np.random.choice(
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
        size=num_samples,
        p=[0.20, 0.25, 0.35, 0.20]
    )

    resting_ecg = np.random.choice(
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
        size=num_samples,
        p=[0.60, 0.25, 0.15]
    )

    exercise_angina = np.random.choice(["No", "Yes"], size=num_samples, p=[0.68, 0.32])

    st_slope = np.random.choice(["Upsloping", "Flat", "Downsloping"], size=num_samples, p=[0.45, 0.43, 0.12])

    # Construct Logistic Risk Function ground truth
    # Log-odds of Cardiac Event Risk
    log_odds = (
        -0.20
        + 0.035 * (age - 50)
        + 0.015 * (resting_bp - 120)
        + 0.008 * (cholesterol - 200)
        + 0.008 * (fasting_bs - 100)
        + 0.045 * (bmi - 25)
        + 0.25 * (hba1c - 5.5)
        - 0.020 * (max_hr - 150)
        + 0.50 * oldpeak
        + np.where(sex == "Male", 0.35, 0.0)
        + np.where(chest_pain_type == "Typical Angina", 0.75, np.where(chest_pain_type == "Asymptomatic", 0.40, 0.0))
        + np.where(exercise_angina == "Yes", 0.85, 0.0)
        + np.where(st_slope == "Flat", 0.55, np.where(st_slope == "Downsloping", 1.10, 0.0))
        + np.where(resting_ecg != "Normal", 0.30, 0.0)
        + np.random.normal(0, 0.35, num_samples) # Unobserved variation
    )

    risk_prob = 1 / (1 + np.exp(-log_odds))
    cardiac_risk = (risk_prob > 0.50).astype(int)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "fasting_bs": fasting_bs,
        "resting_ecg": resting_ecg,
        "max_hr": max_hr,
        "exercise_angina": exercise_angina,
        "oldpeak": oldpeak,
        "st_slope": st_slope,
        "bmi": bmi,
        "hba1c": hba1c,
        config.TARGET_COL: cardiac_risk
    })

    # Introduce ~2.5% random missing values to test Imputer pipelines
    mask_bp = np.random.rand(num_samples) < 0.025
    mask_chol = np.random.rand(num_samples) < 0.025
    mask_bmi = np.random.rand(num_samples) < 0.020
    df.loc[mask_bp, "resting_bp"] = np.nan
    df.loc[mask_chol, "cholesterol"] = np.nan
    df.loc[mask_bmi, "bmi"] = np.nan

    return df

def save_synthetic_dataset(output_path: Path = config.RAW_DATA_PATH):
    df = generate_synthetic_medical_data()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Successfully generated and saved {len(df)} synthetic medical records to {output_path}")
    return df

if __name__ == "__main__":
    save_synthetic_dataset()
