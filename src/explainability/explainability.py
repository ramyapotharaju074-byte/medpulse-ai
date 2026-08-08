import numpy as np
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
import joblib
import shap
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def get_global_feature_importance(model, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Retrieves global feature importance using model attributes or SHAP summary values.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    elif hasattr(model, 'calibrated_classifiers_'):
        # For CalibratedClassifierCV wrapper
        base_clf = model.calibrated_classifiers_[0].estimator
        if hasattr(base_clf, 'coef_'):
            importances = np.abs(base_clf.coef_[0])
        elif hasattr(base_clf, 'feature_importances_'):
            importances = base_clf.feature_importances_
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    total = np.sum(importances) if np.sum(importances) > 0 else 1.0
    importances = importances / total

    # Aggregate one-hot categories back to raw feature names
    feature_impacts = {}
    for name, imp in zip(feature_names, importances):
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
    Computes authentic SHAP local feature attribution values using the SHAP library.
    """
    # Transform raw patient DataFrame
    patient_transformed = preprocessor.transform(patient_df)
    risk_proba = float(model.predict_proba(patient_transformed)[0, 1])

    # Initialize authentic SHAP Explainer
    try:
        if hasattr(model, 'tree_explanation_') or 'RandomForest' in type(model).__name__ or 'GradientBoosting' in type(model).__name__:
            explainer = shap.TreeExplainer(model)
            shap_output = explainer.shap_values(patient_transformed)
        else:
            # Linear / Calibrated / General Classifier Explainer
            explainer = shap.Explainer(model.predict_proba, np.zeros((1, len(feature_names))))
            shap_output = explainer(patient_transformed)
            if hasattr(shap_output, 'values'):
                shap_output = shap_output.values

        # Format SHAP values array for binary positive class (class 1)
        if isinstance(shap_output, list):
            # List of arrays [class_0_shap, class_1_shap]
            shap_vec = shap_output[1][0] if len(shap_output) > 1 else shap_output[0][0]
        elif isinstance(shap_output, np.ndarray):
            if shap_output.ndim == 3:
                # Shape (samples, features, classes)
                shap_vec = shap_output[0, :, 1]
            elif shap_output.ndim == 2:
                shap_vec = shap_output[0]
            else:
                shap_vec = shap_output
        else:
            shap_vec = np.zeros(len(feature_names))

    except Exception as e:
        print(f"[!] SHAP calculation fallback notice: {e}")
        # Robust fallback using feature weights
        if hasattr(model, 'coef_'):
            shap_vec = model.coef_[0] * patient_transformed[0]
        else:
            shap_vec = np.zeros(len(feature_names))

    # Map SHAP values back to original feature groups
    feature_shap_map = {}
    for feat_name, shap_val in zip(feature_names, shap_vec):
        base_feature = feat_name
        for orig in config.FEATURE_COLUMNS:
            if feat_name.startswith(f"cat__{orig}") or feat_name == orig or feat_name.startswith(f"num__{orig}"):
                base_feature = orig
                break
        feature_shap_map[base_feature] = feature_shap_map.get(base_feature, 0.0) + float(shap_val)

    # Build detailed feature driver objects
    contributions = []
    for orig_feat in config.FEATURE_COLUMNS:
        val = patient_df[orig_feat].iloc[0]
        display_name = config.FEATURE_DISPLAY_NAMES.get(orig_feat, orig_feat)
        shap_val = feature_shap_map.get(orig_feat, 0.0)

        contributions.append({
            "feature": orig_feat,
            "display_name": display_name,
            "value": str(val) if pd.notna(val) else "N/A",
            "impact": round(float(shap_val), 4),
            "effect": "Increases Risk" if shap_val > 0 else "Decreases Risk"
        })

    # Sort by absolute SHAP impact magnitude
    contributions = sorted(contributions, key=lambda x: abs(x["impact"]), reverse=True)

    # Categorize Risk Level
    if risk_proba < config.RISK_LOW_MAX:
        risk_category = "Low Cardiac Risk"
        risk_color = "#10b981"
    elif risk_proba < config.RISK_MODERATE_MAX:
        risk_category = "Moderate Risk (Requires Monitoring)"
        risk_color = "#f59e0b"
    else:
        risk_category = "High Cardiac Event Risk (Immediate Review)"
        risk_color = "#ef4444"

    return {
        "risk_probability": round(risk_proba, 4),
        "risk_percentage": round(risk_proba * 100, 1),
        "risk_category": risk_category,
        "risk_color": risk_color,
        "top_drivers": contributions[:6],
        "all_contributions": contributions
    }
