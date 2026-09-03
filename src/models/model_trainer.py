import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import joblib
import json
import sys

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.data.data_generator import save_synthetic_dataset
from src.data.preprocessing import prepare_data, get_transformed_feature_names

from sklearn.calibration import CalibratedClassifierCV

def train_and_compare_models(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trains multiple ML algorithms, evaluates 5-fold cross-validation scores,
    and identifies the best champion model.
    """
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42),
        "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42),
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=500, random_state=42),
        "SVC": CalibratedClassifierCV(SVC(C=1.0, kernel='rbf', random_state=42), ensemble=False)
    }

    results = {}
    trained_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n=======================================================")
    print("   TRAINING & EVALUATING MULTI-MODEL PIPELINE          ")
    print("=======================================================")

    best_auc = -1.0
    best_model_name = ""

    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        mean_cv_auc = cv_scores.mean()
        std_cv_auc = cv_scores.std()

        # Fit full train dataset
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Test set probabilities & predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        results[name] = {
            "cv_roc_auc_mean": round(float(mean_cv_auc), 4),
            "cv_roc_auc_std": round(float(std_cv_auc), 4),
            "y_pred_proba": y_pred_proba,
            "y_pred": y_pred
        }

        print(f" -> {name:18s} | 5-Fold CV ROC-AUC: {mean_cv_auc:.4f} (+/- {std_cv_auc:.4f})")

        if mean_cv_auc > best_auc:
            best_auc = mean_cv_auc
            best_model_name = name

    print(f"\n[*] Champion Model Selected: {best_model_name} (ROC-AUC: {best_auc:.4f})")

    # Save trained models
    for name, model in trained_models.items():
        joblib.dump(model, config.MODELS_DIR / f"{name}.joblib")

    # Save champion model link
    joblib.dump(trained_models[best_model_name], config.MODELS_DIR / "champion_model.joblib")
    
    # Save metadata
    meta = {
        "best_model": best_model_name,
        "best_cv_auc": float(best_auc),
        "models_evaluated": list(models.keys())
    }

def run_full_training_pipeline():
    """
    Executes raw data generation, preprocessing, training, and artifact persistence.
    """
    df = save_synthetic_dataset()
    data_dict = prepare_data(df)
    train_results = train_and_compare_models(data_dict)
    
    # Generate evaluation report artifact
    from src.models.evaluator import generate_evaluation_report
    generate_evaluation_report(data_dict["y_test"], train_results["results"])

    # Export train/test raw CSVs for frontend batch demonstration
    data_dict["X_test_raw"].assign(**{config.TARGET_COL: data_dict["y_test"]}).to_csv(
        config.ARTIFACTS_DIR / "test_sample_patients.csv", index=False
    )
    print(f"[+] Exported sample test CSV to {config.ARTIFACTS_DIR / 'test_sample_patients.csv'}")

    return data_dict, train_results

if __name__ == "__main__":
    run_full_training_pipeline()
