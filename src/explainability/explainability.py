import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src import config


def _get_original_feature(feature_name: str) -> str:
    """
    Maps transformed feature names back to their original
    human-readable feature names.
    """
    for original in config.FEATURE_COLUMNS:
        if (
            feature_name == original
            or feature_name.startswith(f"num__{original}")
            or feature_name.startswith(f"cat__{original}")
        ):
            return original

    return feature_name


def _get_shap_explainer(model, background_data, feature_names):
    """
    Creates a SHAP explainer suitable for the trained model.
    SHAP automatically selects the appropriate algorithm where possible.
    """
    return shap.Explainer(
        model,
        background_data,
        feature_names=feature_names
    )


def get_global_feature_importance(
    model,
    feature_names: List[str],
    background_data=None
) -> List[Dict[str, Any]]:
    """
    Computes global feature importance using mean absolute SHAP values.
    """

    if background_data is None:
        raise ValueError(
            "background_data is required for SHAP global explanations."
        )

    explainer = _get_shap_explainer(
        model,
        background_data,
        feature_names
    )

    shap_output = explainer(background_data)

    values = shap_output.values

    # Handle binary/multi-output SHAP shapes.
    if values.ndim == 3:
        values = values[:, :, 1]

    mean_abs_shap = np.abs(values).mean(axis=0)

    total = mean_abs_shap.sum()
    if total > 0:
        normalized = mean_abs_shap / total
    else:
        normalized = mean_abs_shap

    # Group one-hot encoded features back to original features.
    grouped_importance = {}

    for name, importance in zip(feature_names, normalized):
        original_name = _get_original_feature(name)

        grouped_importance[original_name] = (
            grouped_importance.get(original_name, 0.0)
            + float(importance)
        )

    results = [
        {
            "feature": feature,
            "display_name": config.FEATURE_DISPLAY_NAMES.get(
                feature,
                feature
            ),
            "importance": round(value * 100, 2)
        }
        for feature, value in grouped_importance.items()
    ]

    return sorted(
        results,
        key=lambda x: x["importance"],
        reverse=True
    )


def explain_patient_risk(
    patient_df: pd.DataFrame,
    preprocessor,
    model,
    feature_names: List[str],
    background_data=None
) -> Dict[str, Any]:
    """
    Generates a genuine local SHAP explanation for one patient.
    """

    if background_data is None:
        raise ValueError(
            "background_data is required for SHAP explanations."
        )

    # Transform patient data using the same preprocessing pipeline
    # used during model training.
    patient_transformed = preprocessor.transform(patient_df)

    # Predict probability of cardiac risk.
    risk_proba = float(
        model.predict_proba(patient_transformed)[0, 1]
    )

    # Create SHAP explainer.
    explainer = _get_shap_explainer(
        model,
        background_data,
        feature_names
    )

    # Calculate SHAP values for this patient.
    shap_output = explainer(patient_transformed)

    shap_values = shap_output.values

    # Binary classification can return:
    # (samples, features, classes)
    if shap_values.ndim == 3:
        patient_shap = shap_values[0, :, 1]
    else:
        patient_shap = shap_values[0]

    contributions = []

    for index, transformed_name in enumerate(feature_names):

        original_feature = _get_original_feature(
            transformed_name
        )

        impact = float(patient_shap[index])

        # Find the corresponding patient value.
        if original_feature in patient_df.columns:
            raw_value = patient_df[
                original_feature
            ].iloc[0]
        else:
            raw_value = "N/A"

        contributions.append({
            "feature": original_feature,
            "display_name": config.FEATURE_DISPLAY_NAMES.get(
                original_feature,
                original_feature
            ),
            "value": str(raw_value),
            "impact": round(impact, 6),
            "effect": (
                "Increases Risk"
                if impact > 0
                else "Decreases Risk"
            )
        })

    # Group one-hot encoded contributions by original feature.
    grouped = {}

    for item in contributions:

        feature = item["feature"]

        if feature not in grouped:
            grouped[feature] = {
                "feature": feature,
                "display_name": item["display_name"],
                "value": item["value"],
                "impact": 0.0
            }

        grouped[feature]["impact"] += item["impact"]

    contributions = list(grouped.values())

    for item in contributions:
        item["impact"] = round(
            item["impact"],
            6
        )

        item["effect"] = (
            "Increases Risk"
            if item["impact"] > 0
            else "Decreases Risk"
        )

    # Largest absolute SHAP contributions first.
    contributions.sort(
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    # Risk category.
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
        "risk_percentage": round(
            risk_proba * 100,
            1
        ),
        "risk_category": risk_category,
        "risk_color": risk_color,
        "top_drivers": contributions[:6],
        "all_contributions": contributions
    }
