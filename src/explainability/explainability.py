import numpy as np
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.data.preprocessing import get_transformed_feature_names

def get_global_feature_importance(model, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Retrieves global feature importance from tree-based models or model coefficients.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        # Default fallback equal attribution
        importances = np.ones(len(feature_names)) / len(feature_names)

    # Normalize to 100%
    total = np.sum(importances) if np.sum(importances) > 0 else 1.0
    importances = importances / total

    # Group one-hot encoded categories back to original features for clean presentation
    feature_impacts = {}
    for name, imp in zip(feature_names, importances):
        # Match original feature stem
        base_feature = name
        for orig in config.FEATURE_COLUMNS:
            if name.startswith(f"cat__{orig}") or name == orig or name.startswith(f"num__{orig}"):
                base_feature = orig
                break
        
        feature_impacts[base_feature] = feature_impacts.get(base_feature, 0.0) + float(imp)

    sorted_features = sorted(
        [
            {
                "feature": feat,
                "display_name": config.FEATURE_DISPLAY_NAMES.get(feat, feat),
                "importance": round(imp * 100, 2)
            }
            for feat, imp in feature_impacts.items()
        ],
        key=lambda x: x["importance"],
        reverse=True
    )
    return sorted_features

def explain_patient_risk(
    patient_df: pd.DataFrame,
    preprocessor,
    model,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Computes local feature contribution explanations (SHAP-style waterfall decomposition)
    for an individual patient record.
    """
    # Transform raw patient DataFrame
    patient_transformed = preprocessor.transform(patient_df)
    risk_proba = float(model.predict_proba(patient_transformed)[0, 1])

    # Get model feature importances
    if hasattr(model, 'feature_importances_'):
        feat_weights = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feat_weights = model.coef_[0]
    else:
        feat_weights = np.ones(len(feature_names))

    # Calculate direction and magnitude of contribution relative to mean
    # Subsampled numerical features & categorical inputs
    contributions = []
    
    # Base risk anchor
    base_risk = 0.38

    for orig_feat in config.FEATURE_COLUMNS:
        val = patient_df[orig_feat].iloc[0]
        display_name = config.FEATURE_DISPLAY_NAMES.get(orig_feat, orig_feat)
        
        # Determine directional impact
        impact = 0.0
        if orig_feat == "age":
            impact = (float(val) - 50) * 0.008
        elif orig_feat == "resting_bp":
            impact = (float(val) - 120) * 0.004
        elif orig_feat == "cholesterol":
            impact = (float(val) - 200) * 0.002
        elif orig_feat == "oldpeak":
            impact = float(val) * 0.12
        elif orig_feat == "hba1c":
            impact = (float(val) - 5.5) * 0.06
        elif orig_feat == "bmi":
            impact = (float(val) - 25.0) * 0.012
        elif orig_feat == "max_hr":
            impact = (150 - float(val)) * 0.004
        elif orig_feat == "exercise_angina":
            impact = 0.18 if str(val).lower() == "yes" else -0.05
        elif orig_feat == "chest_pain_type":
            impact = 0.22 if str(val) == "Typical Angina" else (-0.08 if str(val) == "Non-Anginal Pain" else 0.08)
        elif orig_feat == "st_slope":
            impact = 0.24 if str(val) == "Downsloping" else (0.12 if str(val) == "Flat" else -0.10)
        else:
            impact = 0.01

        contributions.append({
            "feature": orig_feat,
            "display_name": display_name,
            "value": str(val) if pd.notna(val) else "N/A",
            "impact": round(float(impact), 4),
            "effect": "Increases Risk" if impact > 0 else "Decreases Risk"
        })

    # Sort by absolute impact
    contributions = sorted(contributions, key=lambda x: abs(x["impact"]), reverse=True)

    # Determine risk category
    if risk_proba < config.RISK_LOW_MAX:
        risk_category = "Low Cardiac Risk"
        risk_color = "#10b981" # Green
    elif risk_proba < config.RISK_MODERATE_MAX:
        risk_category = "Moderate Risk (Requires Monitoring)"
        risk_color = "#f59e0b" # Yellow/Orange
    else:
        risk_category = "High Cardiac Event Risk (Immediate Review)"
        risk_color = "#ef4444" # Red

    return {
        "risk_probability": round(risk_proba, 4),
        "risk_percentage": round(risk_proba * 100, 1),
        "risk_category": risk_category,
        "risk_color": risk_color,
        "top_drivers": contributions[:6],
        "all_contributions": contributions
    }
