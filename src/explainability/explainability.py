import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import shap


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src import config


# ============================================================
# FEATURE NAME MAPPING
# ============================================================

def _get_original_feature(feature_name: str) -> str:
    """
    Maps transformed feature names back to their
    original human-readable feature names.
    """

    for original in config.FEATURE_COLUMNS:

        if feature_name == original:
            return original

        if feature_name.startswith(f"num__{original}"):
            return original

        if feature_name.startswith(f"cat__{original}"):
            return original

    return feature_name


# ============================================================
# SHAP EXPLAINER
# ============================================================

def _get_shap_explainer(
    model,
    background_data,
    feature_names
):
    """
    Creates a SHAP explainer compatible with different
    classification models.

    Tree-based models:
        Uses TreeExplainer.

    Other models such as:
        MLPClassifier
        LogisticRegression
        SVC

    are explained through model.predict_proba().
    """

    if background_data is None:
        raise ValueError(
            "background_data is required for SHAP explanations."
        )

    background_data = np.asarray(background_data)

    # --------------------------------------------------------
    # TREE-BASED MODELS
    # --------------------------------------------------------

    tree_model_names = {
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "LGBMClassifier",
        "CatBoostClassifier"
    }

    model_name = model.__class__.__name__

    if model_name in tree_model_names:

        try:
            return shap.TreeExplainer(model)
        except Exception:
            pass

    # --------------------------------------------------------
    # GENERAL CLASSIFICATION MODELS
    # --------------------------------------------------------
    # This includes MLPClassifier.
    #
    # SHAP receives a callable function rather than the model
    # object directly.

    def predict_function(X):
        X = np.asarray(X)

        probabilities = model.predict_proba(X)

        # Return probability of positive/risk class
        return probabilities[:, 1]

    return shap.Explainer(
        predict_function,
        background_data,
        feature_names=feature_names
    )


# ============================================================
# SAFE SHAP VALUE EXTRACTION
# ============================================================

def _extract_shap_values(
    shap_output,
    class_index: int = 1
):
    """
    Safely extracts SHAP values.

    Supported formats:

        (samples, features)

        (samples, features, classes)

        list of arrays

        (features,)
    """

    values = shap_output.values

    # --------------------------------------------------------
    # LIST FORMAT
    # --------------------------------------------------------

    if isinstance(values, list):

        if len(values) > class_index:
            values = values[class_index]
        else:
            values = values[0]

    values = np.asarray(values)

    # --------------------------------------------------------
    # 3D FORMAT
    # --------------------------------------------------------

    if values.ndim == 3:

        return values[:, :, class_index]

    # --------------------------------------------------------
    # 2D FORMAT
    # --------------------------------------------------------

    if values.ndim == 2:

        return values

    # --------------------------------------------------------
    # 1D FORMAT
    # --------------------------------------------------------

    if values.ndim == 1:

        return values.reshape(1, -1)

    raise ValueError(
        f"Unexpected SHAP output shape: {values.shape}"
    )


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

def get_global_feature_importance(
    model,
    preprocessor=None,
    feature_names: List[str] = None,
    background_data=None
) -> List[Dict[str, Any]]:
    """
    Computes global feature importance using mean absolute
    SHAP values.

    Parameters
    ----------
    model:
        Trained classification model.

    preprocessor:
        Training preprocessing pipeline.

    feature_names:
        Names of transformed features.

    background_data:
        Transformed background dataset.
    """

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if background_data is None:

        raise ValueError(
            "background_data is required for SHAP "
            "global explanations."
        )

    if feature_names is None:

        raise ValueError(
            "feature_names is required for SHAP "
            "global explanations."
        )

    if model is None:

        raise ValueError(
            "model is required for SHAP "
            "global explanations."
        )

    # --------------------------------------------------------
    # CREATE SHAP EXPLAINER
    # --------------------------------------------------------

    explainer = _get_shap_explainer(
        model=model,
        background_data=background_data,
        feature_names=feature_names
    )

    # --------------------------------------------------------
    # CALCULATE SHAP VALUES
    # --------------------------------------------------------

    shap_output = explainer(
        np.asarray(background_data)
    )

    values = _extract_shap_values(
        shap_output,
        class_index=1
    )

    # --------------------------------------------------------
    # MEAN ABSOLUTE SHAP
    # --------------------------------------------------------

    mean_abs_shap = np.abs(values).mean(axis=0)

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    total = float(
        mean_abs_shap.sum()
    )

    if total > 0:

        normalized = (
            mean_abs_shap / total
        )

    else:

        normalized = mean_abs_shap

    # --------------------------------------------------------
    # GROUP TRANSFORMED FEATURES
    # --------------------------------------------------------

    grouped_importance = {}

    for name, importance in zip(
        feature_names,
        normalized
    ):

        original_name = _get_original_feature(
            name
        )

        grouped_importance[
            original_name
        ] = (
            grouped_importance.get(
                original_name,
                0.0
            )
            + float(importance)
        )

    # --------------------------------------------------------
    # CREATE RESPONSE
    # --------------------------------------------------------

    results = []

    for feature, value in grouped_importance.items():

        results.append({

            "feature": feature,

            "display_name": (
                config.FEATURE_DISPLAY_NAMES.get(
                    feature,
                    feature
                )
            ),

            "importance": round(
                float(value) * 100,
                2
            )
        })

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    return results


# ============================================================
# LOCAL PATIENT RISK EXPLANATION
# ============================================================

def explain_patient_risk(
    patient_df: pd.DataFrame,
    preprocessor,
    model,
    feature_names: List[str],
    background_data=None
) -> Dict[str, Any]:
    """
    Generates a local SHAP explanation for one patient.
    """

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if background_data is None:

        raise ValueError(
            "background_data is required for SHAP "
            "patient explanations."
        )

    if patient_df is None or patient_df.empty:

        raise ValueError(
            "patient_df cannot be empty."
        )

    if preprocessor is None:

        raise ValueError(
            "preprocessor is required."
        )

    if model is None:

        raise ValueError(
            "model is required."
        )

    if feature_names is None:

        raise ValueError(
            "feature_names is required."
        )

    # --------------------------------------------------------
    # TRANSFORM PATIENT DATA
    # --------------------------------------------------------

    patient_transformed = (
        preprocessor.transform(
            patient_df
        )
    )

    patient_transformed = np.asarray(
        patient_transformed
    )

    # --------------------------------------------------------
    # PREDICT CARDIAC RISK
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        patient_transformed
    )

    risk_proba = float(
        probabilities[0, 1]
    )

    # --------------------------------------------------------
    # SHAP EXPLAINER
    # --------------------------------------------------------

    explainer = _get_shap_explainer(
        model=model,
        background_data=background_data,
        feature_names=feature_names
    )

    # --------------------------------------------------------
    # CALCULATE PATIENT SHAP VALUES
    # --------------------------------------------------------

    shap_output = explainer(
        patient_transformed
    )

    patient_values = _extract_shap_values(
        shap_output,
        class_index=1
    )

    patient_shap = patient_values[0]

    # --------------------------------------------------------
    # CREATE CONTRIBUTIONS
    # --------------------------------------------------------

    contributions = []

    for index, transformed_name in enumerate(
        feature_names
    ):

        # Safety check
        if index >= len(patient_shap):

            break

        # ----------------------------------------------------
        # ORIGINAL FEATURE
        # ----------------------------------------------------

        original_feature = (
            _get_original_feature(
                transformed_name
            )
        )

        # ----------------------------------------------------
        # SHAP IMPACT
        # ----------------------------------------------------

        impact = float(
            patient_shap[index]
        )

        # ----------------------------------------------------
        # ORIGINAL PATIENT VALUE
        # ----------------------------------------------------

        if original_feature in patient_df.columns:

            raw_value = patient_df[
                original_feature
            ].iloc[0]

        else:

            raw_value = "N/A"

        # ----------------------------------------------------
        # EFFECT
        # ----------------------------------------------------

        if impact > 0:

            effect = "Increases Risk"

        elif impact < 0:

            effect = "Decreases Risk"

        else:

            effect = "No Significant Effect"

        # ----------------------------------------------------
        # ADD CONTRIBUTION
        # ----------------------------------------------------

        contributions.append({

            "feature": original_feature,

            "display_name": (
                config.FEATURE_DISPLAY_NAMES.get(
                    original_feature,
                    original_feature
                )
            ),

            "value": str(
                raw_value
            ),

            "impact": round(
                impact,
                6
            ),

            "effect": effect
        })

    # ========================================================
    # GROUP ONE-HOT FEATURES
    # ========================================================

    grouped = {}

    for item in contributions:

        feature = item["feature"]

        if feature not in grouped:

            grouped[feature] = {

                "feature": feature,

                "display_name": (
                    item["display_name"]
                ),

                "value": item["value"],

                "impact": 0.0
            }

        grouped[
            feature
        ]["impact"] += (
            item["impact"]
        )

    contributions = list(
        grouped.values()
    )

    # ========================================================
    # UPDATE EFFECT AFTER GROUPING
    # ========================================================

    for item in contributions:

        item["impact"] = round(
            float(item["impact"]),
            6
        )

        if item["impact"] > 0:

            item["effect"] = (
                "Increases Risk"
            )

        elif item["impact"] < 0:

            item["effect"] = (
                "Decreases Risk"
            )

        else:

            item["effect"] = (
                "No Significant Effect"
            )

    # ========================================================
    # SORT BY ABSOLUTE IMPACT
    # ========================================================

    contributions.sort(
        key=lambda x: abs(
            x["impact"]
        ),
        reverse=True
    )

    # ========================================================
    # RISK CATEGORY
    # ========================================================

    if risk_proba < config.RISK_LOW_MAX:

        risk_category = (
            "Low Cardiac Risk"
        )

        risk_color = "#10b981"

    elif risk_proba < config.RISK_MODERATE_MAX:

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

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "risk_probability": round(
            risk_proba,
            4
        ),

        "risk_percentage": round(
            risk_proba * 100,
            1
        ),

        "risk_category": risk_category,

        "risk_color": risk_color,

        "top_drivers": (
            contributions[:6]
        ),

        "all_contributions": (
            contributions
        )
    }