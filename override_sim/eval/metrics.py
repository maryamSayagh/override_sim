
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan,
        "pr_auc": average_precision_score(y, p) if np.unique(y).size == 2 else np.nan,
        "brier": brier_score_loss(y, p),
        "accuracy": accuracy_score(y, pred),
        "pred_mean": float(p.mean()),
        "label_mean": float(y.mean()),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def probability_recovery(p_override_true, y_proba):
    return {
        "mae": mean_absolute_error(p_override_true, y_proba),
        "rmse": np.sqrt(mean_squared_error(p_override_true, y_proba)),
    }


def evaluation_report(override, y_proba, p_override_true=None):
    results = {
        "prediction": compute_metrics(override, y_proba)
    }

    if p_override_true is not None:
        results["probability_recovery"] = probability_recovery(
            p_override_true, y_proba
        )

    return results






"""
for usuage, I'll keep this here to edit after adding the rest of evaluation methods,
throughout the evaluation folder.


results = evaluation_report(
    override=df_test["override"],
    y_proba=model.predict_proba(X_test)[:, 1],
    p_override_true=df_test["p_override_true"],
)
"""
