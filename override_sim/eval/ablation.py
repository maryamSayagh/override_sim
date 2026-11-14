""" in the ablation evaluation method, we measure how each 
group of features impact the global data. 

The groups we are evaluation so far are (referred to here as "feature_group": 
  - Utility dominant world: "U": ["urgency", ...],
  - Behavioral/trust dominant world: "B": ["B_ovr_ewma", ...],
  - Incentives dominant world:  "I": ["incentive", ...],

Metrics used:
    - ROC-AUC
    - PR-AUC
    - Brier score
    - Performance differences relative to the full model

Within-world and cross-world generalization are handled
separately in generalization.py.

"""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


def evaluate_ablation(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_groups,
):
   
   #  Train the full model and leave-one-group-out variants.
  
    results = []

    experiments = {
        "full": [],
        **feature_groups,
    }

    for group, excluded in experiments.items():

        features = [
            col for col in X_train.columns
            if col not in excluded
        ]

        if not features:
            raise ValueError(
                f"No features remaining after removing {group}"
            )

        estimator = clone(model)
        estimator.fit(X_train[features], y_train)

        probabilities = estimator.predict_proba(
            X_test[features]
        )[:, 1]

        metrics = {
            "removed_group": group,
            "n_features": len(features),
            "roc_auc": (
                roc_auc_score(y_test, probabilities)
                if np.unique(y_test).size == 2
                else np.nan
            ),
            "pr_auc": (
                average_precision_score(y_test, probabilities)
                if np.unique(y_test).size == 2
                else np.nan
            ),
            "brier": brier_score_loss(
                y_test, probabilities
            ),
        }

        results.append(metrics)

    df = pd.DataFrame(results)

    # Full-model reference
    baseline = df.loc[
        df["removed_group"] == "full"
    ].iloc[0]

    # Positive delta means removing a group worsened performance
    df["delta_auc"] = (
        baseline["roc_auc"] - df["roc_auc"]
    )

    df["delta_pr_auc"] = (
        baseline["pr_auc"] - df["pr_auc"]
    )

    df["delta_brier"] = (
        df["brier"] - baseline["brier"]
    )

    return df


def summarize_ablation(results):
    """
    Return feature-group contributions relative to
    the full-model baseline.
    """
    return (
        results.loc[
            results["removed_group"] != "full",
            [
                "removed_group",
                "n_features",
                "delta_auc",
                "delta_pr_auc",
                "delta_brier",
            ],
        ]
        .sort_values(
            "delta_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )


""" usuage call, storing it here so I can add it later to a reporting modul 

from eval.ablation import evaluate_ablation, summarize_ablation

feature_groups = {
    "B": ["B_ovr_ewma", "B_ovr_rolling10"],
    "U": ["urgency"],
    "I": ["incentive"],
}

results = evaluate_ablation(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_groups,
)

print(summarize_ablation(results))"""
