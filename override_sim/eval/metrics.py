from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def compute_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "pred_mean": float(np.mean(p)),
        "label_mean": float(np.mean(y_true)),
    }
