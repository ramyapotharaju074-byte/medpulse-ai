import numpy as np
import pandas as pd
import shap

from typing import Dict, Any, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src import config


# ============================================================
# Feature Name Mapping
# ============================================================

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


# ============================================================
# Global Feature Importance
# ============================================================

def get_global_feature_importance(
    model,
    feature_names: List[str],
    background_data=None
) -> List[Dict[str, Any]]:
    """
    Calculates global feature importance using the trained model.

    Supports tree-based models, linear models, and calibrated
    classifiers.

    background_data is accepted for compatibility with the
    explainability API.
    """

    # --------------------------------------------------------
    # Obtain model feature importance
    # --------------------------------------------------------

    if hasattr(model, "feature_importances_"):

        importances = np.asarray(
            model.feature_importances_
        )

    elif hasattr(model, "coef_"):

        importances = np.abs(
            np.asarray(model.coef_[0])
        )

    elif hasattr(model, "calibrated_classifiers_"):

        base_clf = (
            model.calibrated_classifiers_[0].estimator
        )

        if hasattr(base_clf, "coef_"):

            importances = np.abs(
                np.asarray(base_clf.coef_[0])
            )

        elif hasattr(base_clf, "feature_importances_"):

            importances = np.asarray(
                base_clf.feature_importances_
            )

        else:

            importances = np.ones(
                len(feature_names)
            )

    else:

        importances = np.ones(
            len(feature_names)
        )

    # --------------------------------------------------------
    # Make sure feature count matches
    # --------------------------------------------------------

    if len(importances) != len(feature_names):

        importances = np.resize(
            importances,
            len(feature_names)
        )

    # --------------------------------------------------------
    # Normalize importance
    # --------------------------------------------------------

    total = np.sum(importances)

    if total > 0:

        importances = importances / total

    # --------------------------------------------------------
    # Group one-hot encoded features
    # --------------------------------------------------------

    feature_impacts = {}

    for name, importance in zip(
        feature_names,
        importances
    ):

        base_feature = _get_original_feature(
            name
        )

        feature_impacts[base_feature] = (
            feature_impacts.get(
                base_feature,
                0.0
            )
            + float(importance)
        )

    # --------------------------------------------------------
    # Build response
    # --------------------------------------------------------

    results = [

        {
            "feature": feature,

            "display_name":
                config.FEATURE_DISPLAY_NAMES.get(
                    feature,
                    feature
                ),

            "importance":
                round(
                    importance * 100,
                    2
                )
        }

        for feature, importance
        in feature_impacts.items()
    ]

    # --------------------------------------------------------
    # Sort descending
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    return results


# ============================================================
# Local SHAP Explanation
# ============================================================

def explain_patient_risk(
    patient_df: pd.DataFrame,
    preprocessor,
    model,
    feature_names: List[str],
    background_data=None
) -> Dict[str, Any]:
    """
    Computes a local SHAP explanation for one patient.

    background_data must be in the same transformed feature
    space as the model input.
    """

    # --------------------------------------------------------
    # Validate background data
    # --------------------------------------------------------

    if background_data is None:

        raise ValueError(
            "SHAP background data is required."
        )

    # --------------------------------------------------------
    # Transform patient data
    # --------------------------------------------------------

    patient_transformed = (
        preprocessor.transform(
            patient_df
        )
    )

    # --------------------------------------------------------
    # Predict risk probability
    # --------------------------------------------------------

    try:

        risk_proba = float(
            model.predict_proba(
                patient_transformed
            )[0, 1]
        )

    except Exception as e:

        raise ValueError(
            f"Model prediction failed: {str(e)}"
        )

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    try:

        model_name = type(model).__name__

        # ----------------------------------------------------
        # Tree-based models
        # ----------------------------------------------------

        if (
            hasattr(
                model,
                "feature_importances_"
            )
            or "RandomForest" in model_name
            or "GradientBoosting" in model_name
            or "XGB" in model_name
            or "LGBM" in model_name
        ):

            explainer = shap.TreeExplainer(
                model
            )

            shap_output = explainer.shap_values(
                patient_transformed
            )

        # ----------------------------------------------------
        # General / Linear / Calibrated models
        # ----------------------------------------------------

        else:

            explainer = shap.Explainer(
                model.predict_proba,
                background_data
            )

            shap_output = explainer(
                patient_transformed
            )

            if hasattr(
                shap_output,
                "values"
            ):

                shap_output = (
                    shap_output.values
                )

        # ----------------------------------------------------
        # Normalize SHAP output shape
        # ----------------------------------------------------

        if isinstance(
            shap_output,
            list
        ):

            # Older SHAP binary classification format
            if len(shap_output) > 1:

                shap_vec = (
                    shap_output[1][0]
                )

            else:

                shap_vec = (
                    shap_output[0][0]
                )

        elif isinstance(
            shap_output,
            np.ndarray
        ):

            if shap_output.ndim == 3:

                # samples, features, classes
                shap_vec = (
                    shap_output[0, :, 1]
                )

            elif shap_output.ndim == 2:

                shap_vec = (
                    shap_output[0]
                )

            else:

                shap_vec = shap_output

        else:

            shap_vec = np.zeros(
                len(feature_names)
            )

    except Exception as e:

        print(
            f"[!] SHAP calculation fallback: {e}"
        )

        # ----------------------------------------------------
        # Fallback for linear models
        # ----------------------------------------------------

        if hasattr(model, "coef_"):

            shap_vec = (
                np.asarray(
                    model.coef_[0]
                )
                * np.asarray(
                    patient_transformed[0]
                )
            )

        else:

            shap_vec = np.zeros(
                len(feature_names)
            )

    # --------------------------------------------------------
    # Make sure SHAP vector has correct length
    # --------------------------------------------------------

    shap_vec = np.asarray(
        shap_vec
    ).flatten()

    if len(shap_vec) != len(feature_names):

        shap_vec = np.resize(
            shap_vec,
            len(feature_names)
        )

    # --------------------------------------------------------
    # Group SHAP values by original feature
    # --------------------------------------------------------

    feature_shap_map = {}

    for feature_name, shap_value in zip(
        feature_names,
        shap_vec
    ):

        original_feature = (
            _get_original_feature(
                feature_name
            )
        )

        feature_shap_map[
            original_feature
        ] = (
            feature_shap_map.get(
                original_feature,
                0.0
            )
            + float(shap_value)
        )

    # --------------------------------------------------------
    # Build patient contribution list
    # --------------------------------------------------------

    contributions = []

    for original_feature in (
        config.FEATURE_COLUMNS
    ):

        if original_feature in patient_df.columns:

            value = patient_df[
                original_feature
            ].iloc[0]

        else:

            value = "N/A"

        shap_value = feature_shap_map.get(
            original_feature,
            0.0
        )

        display_name = (
            config.FEATURE_DISPLAY_NAMES.get(
                original_feature,
                original_feature
            )
        )

        contributions.append(
            {
                "feature": original_feature,

                "display_name": display_name,

                "value": (
                    str(value)
                    if pd.notna(value)
                    else "N/A"
                ),

                "impact": round(
                    float(shap_value),
                    6
                ),

                "effect": (
                    "Increases Risk"
                    if shap_value > 0
                    else "Decreases Risk"
                    if shap_value < 0
                    else"No Significant Impact"
                )
            }
        )

    # --------------------------------------------------------
    # Sort by absolute SHAP impact
    # --------------------------------------------------------

    contributions.sort(
        key=lambda x: abs(
            x["impact"]
        ),
        reverse=True
    )

    # --------------------------------------------------------
    # Risk Category
    # --------------------------------------------------------

    if (
        risk_proba
        < config.RISK_LOW_MAX
    ):

        risk_category = (
            "Low Cardiac Risk"
        )

        risk_color = "#10b981"

    elif (
        risk_proba
        < config.RISK_MODERATE_MAX
    ):

        risk_category = (
            "Moderate Risk "
            "(Requires Monitoring)"
        )

        risk_color = "#f59e0b"

    else:

        risk_category = (
            "High Cardiac Event Risk "
            "(Immediate Review)"
        )

        risk_color = "#ef4444"

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {

        "risk_probability":
            round(
                risk_proba,
                4
            ),

        "risk_percentage":
            round(
                risk_proba * 100,
                1
            ),

        "risk_category":
            risk_category,

        "risk_color":
            risk_color,

        "top_drivers":
            contributions[:6],

        "all_contributions":
            contributions
    }