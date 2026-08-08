import numpy as np
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
import json
import sys

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def compute_model_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    """
    Calculates comprehensive classification metrics and curve data points for frontend display.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, roc_thresh = roc_curve(y_true, y_pred_proba)
    prec_pts, rec_pts, pr_thresh = precision_recall_curve(y_true, y_pred_proba)

    # Subsample curve data points for fast JSON API transmission
    subsample_idx = np.linspace(0, len(fpr) - 1, min(40, len(fpr))).astype(int)

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "brier_score": round(float(brier), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
        },
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr[subsample_idx]],
            "tpr": [round(float(x), 4) for x in tpr[subsample_idx]]
        },
        "pr_curve": {
            "precision": [round(float(x), 4) for x in prec_pts[::max(1, len(prec_pts)//30)]],
            "recall": [round(float(x), 4) for x in rec_pts[::max(1, len(rec_pts)//30)]]
        }
    }

def generate_evaluation_report(y_true: np.ndarray, model_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compiles full comparison metrics report across all trained models.
    """
    report = {}
    summary_table = []

    for name, res in model_results.items():
        metrics = compute_model_metrics(y_true, res["y_pred_proba"])
        report[name] = metrics
        
        summary_table.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "brier_score": metrics["brier_score"],
            "cv_auc_mean": res.get("cv_roc_auc_mean", 0.0)
        })

    report_data = {
        "model_metrics": report,
        "summary_table": sorted(summary_table, key=lambda x: x["roc_auc"], reverse=True)
    }

    report_path = config.ARTIFACTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"[+] Saved evaluation report to {report_path}")

    return report_data
