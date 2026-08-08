import numpy as np
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
from scipy.stats import ks_2samp
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) between baseline and current distributions.
    PSI < 0.1: No significant distribution change.
    0.1 <= PSI < 0.2: Moderate shift.
    PSI >= 0.2: Significant drift (Action required).
    """
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(baseline, percentiles)
    buckets[0] -= 1e-5
    buckets[-1] += 1e-5

    baseline_counts, _ = np.histogram(baseline, bins=buckets)
    current_counts, _ = np.histogram(current, bins=buckets)

    baseline_pct = np.where(baseline_counts == 0, 0.0001, baseline_counts) / len(baseline)
    current_pct = np.where(current_counts == 0, 0.0001, current_counts) / len(current)

    psi_val = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi_val)

def analyze_dataset_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compares baseline training features against production sample data to detect Data Drift.
    """
    feature_drift_results = []
    high_drift_count = 0

    for col in config.NUMERICAL_FEATURES:
        if col not in baseline_df.columns or col not in current_df.columns:
            continue

        base_vals = baseline_df[col].dropna().values
        curr_vals = current_df[col].dropna().values

        # Compute KS-test
        ks_stat, p_value = ks_2samp(base_vals, curr_vals)
        psi_score = calculate_psi(base_vals, curr_vals)

        # Classify drift level
        if psi_score >= 0.20 or p_value < 0.01:
            drift_status = "High Drift (Alert)"
            status_color = "#ef4444"
            high_drift_count += 1
        elif psi_score >= 0.10 or p_value < 0.05:
            drift_status = "Moderate Shift"
            status_color = "#f59e0b"
        else:
            drift_status = "Stable (No Drift)"
            status_color = "#10b981"

        feature_drift_results.append({
            "feature": col,
            "display_name": config.FEATURE_DISPLAY_NAMES.get(col, col),
            "psi_score": round(psi_score, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "status": drift_status,
            "status_color": status_color,
            "baseline_mean": round(float(np.mean(base_vals)), 2),
            "current_mean": round(float(np.mean(curr_vals)), 2)
        })

    # Overall Pipeline Drift Status
    if high_drift_count > 0:
        overall_status = "DRIFT DETECTED - RETRAINING RECOMMENDED"
        overall_color = "#ef4444"
    else:
        overall_status = "PIPELINE HEALTHY - DATA DISTRIBUTIONS STABLE"
        overall_color = "#10b981"

    return {
        "overall_status": overall_status,
        "overall_color": overall_color,
        "features_analyzed": len(feature_drift_results),
        "high_drift_features_count": high_drift_count,
        "feature_metrics": feature_drift_results
    }
