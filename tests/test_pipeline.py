import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to sys path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src import config
from src.data.data_generator import generate_synthetic_medical_data
from src.data.preprocessing import prepare_data, build_preprocessor_pipeline
from src.models.model_trainer import train_and_compare_models
from src.models.evaluator import compute_model_metrics
from src.explainability.explainability import explain_patient_risk, get_global_feature_importance
from src.monitoring.drift_monitor import calculate_psi, analyze_dataset_drift

def test_data_generation():
    df = generate_synthetic_medical_data(num_samples=100, random_seed=42)
    assert len(df) == 100
    assert config.TARGET_COL in df.columns
    for feat in config.FEATURE_COLUMNS:
        assert feat in df.columns

def test_preprocessing_pipeline():
    df = generate_synthetic_medical_data(num_samples=150, random_seed=42)
    data_dict = prepare_data(df, test_size=0.20)
    
    assert data_dict["X_train"].shape[0] == 120
    assert data_dict["X_test"].shape[0] == 30
    assert len(data_dict["y_train"]) == 120
    assert len(data_dict["y_test"]) == 30
    assert data_dict["preprocessor"] is not None

def test_model_training_and_selection():
    df = generate_synthetic_medical_data(num_samples=200, random_seed=42)
    data_dict = prepare_data(df)
    results = train_and_compare_models(data_dict)
    
    assert "Champion" in results["best_model_name"] or len(results["best_model_name"]) > 0
    assert results["best_model_name"] in results["trained_models"]
    
    # Check predictions format
    model = results["trained_models"][results["best_model_name"]]
    preds = model.predict(data_dict["X_test"])
    assert len(preds) == 40

def test_evaluation_metrics():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_probas = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.85, 0.15])
    
    metrics = compute_model_metrics(y_true, y_probas)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["tp"] == 4
    assert metrics["confusion_matrix"]["tn"] == 4

def test_explainability_output():
    df = generate_synthetic_medical_data(num_samples=100, random_seed=42)
    data_dict = prepare_data(df)
    results = train_and_compare_models(data_dict)
    champion = results["trained_models"][results["best_model_name"]]
    
    sample_patient = df[config.FEATURE_COLUMNS].iloc[[0]]
    explanation = explain_patient_risk(
        sample_patient,
        data_dict["preprocessor"],
        champion,
        data_dict["feature_names"]
    )
    
    assert "risk_probability" in explanation
    assert "risk_category" in explanation
    assert len(explanation["top_drivers"]) > 0

def test_drift_monitor_psi():
    base = np.random.normal(100, 15, 500)
    current_stable = np.random.normal(100, 15, 500)
    current_shifted = np.random.normal(130, 20, 500)

    psi_stable = calculate_psi(base, current_stable)
    psi_shifted = calculate_psi(base, current_shifted)

    assert psi_stable < 0.10
    assert psi_shifted > 0.20
