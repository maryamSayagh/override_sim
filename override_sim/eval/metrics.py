"""from __future__ import annotations
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
"""

#extended version :

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    confusion_matrix,
)

def compute_metrics(y_true, y_prob, threshold=0.5):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= threshold).astype(int)

    return {
        "roc_auc": (
            roc_auc_score(y, p)
            if np.unique(y).size == 2 else np.nan
        ),
        "pr_auc": (
            average_precision_score(y, p)
            if np.unique(y).size == 2 else np.nan
        ),
        "brier": brier_score_loss(y, p),
        "accuracy": accuracy_score(y, pred),
        "pred_mean": float(p.mean()),
        "label_mean": float(y.mean()),
        "confusion_matrix": confusion_matrix(
            y, pred, labels=[0, 1]
        ).tolist(),
    }
