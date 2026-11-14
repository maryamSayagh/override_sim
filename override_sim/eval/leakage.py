
"""
 Leakage and Feature Access Validation

Purpose:
    Prevent hidden simulation information, target leakage,
    and train/test contamination from invalidating
    predictive evaluations.

Usage:
    Run before training or evaluating a model.

    Failed checks raise ValueError.
"""

import numpy as np
import pandas as pd


# -------------------------------------------------------
# 1. Forbidden simulator information
# -------------------------------------------------------

FORBIDDEN_COLUMNS = {
    "override",
    "y_true",
    "p_override_true",
    "intercept_b",
    "confusion_type",
    "trust_T_latent",
    "human_prob_latent",
    "ai_correct",
    "ai_error",
    "y_final",
}

FORBIDDEN_PATTERNS = (
    "_latent",
    "_true",
)


def validate_features(X, target="override"):
    """
    Ensure model inputs contain no known forbidden
    simulator variables or target labels.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("Feature dataset is empty.")

    forbidden = set(FORBIDDEN_COLUMNS) | {target}

    leaked = [
        col for col in X.columns
        if col in forbidden
        or col.endswith(FORBIDDEN_PATTERNS)
    ]

    if leaked:
        raise ValueError(
            f"Feature leakage detected: {leaked}"
        )

    if not X.columns.is_unique:
        raise ValueError("Duplicate feature names detected.")

    if X.isna().any().any():
        raise ValueError(
            "Missing values detected. "
            "Use training-fitted preprocessing if needed."
        )

    return True


# -------------------------------------------------------
# 2. Train/test separation
# -------------------------------------------------------

def validate_split(
    train_df,
    test_df,
    id_col="instance_id",
    group_col="user_id",
    require_group_separation=True,
):
    """
    Detect overlapping instances and, optionally,
    overlapping users between training and testing.

    For cross-world experiments, instance IDs must be
    globally unique before enabling ID-overlap checking.
    """

    if id_col is not None:
        for df in (train_df, test_df):
            if id_col not in df.columns:
                raise ValueError(
                    f"Missing required ID column: {id_col}"
                )

            if df[id_col].isna().any():
                raise ValueError("Null instance IDs detected.")

        train_ids = set(train_df[id_col])
        test_ids = set(test_df[id_col])

        overlap = train_ids & test_ids

        if overlap:
            raise ValueError(
                f"Train/test instance overlap: "
                f"{len(overlap)} IDs."
            )

    if require_group_separation:
        for df in (train_df, test_df):
            if group_col not in df.columns:
                raise ValueError(
                    f"Missing group column: {group_col}"
                )

        train_users = set(train_df[group_col].dropna())
        test_users = set(test_df[group_col].dropna())

        overlap = train_users & test_users

        if overlap:
            raise ValueError(
                f"Train/test user overlap: "
                f"{len(overlap)} users."
            )

    return True


# -------------------------------------------------------
# 3. Historical feature checks
# -------------------------------------------------------

HISTORY_FEATURES = {
    "B_ovr_ewma",
    "B_ovr_rolling10",
    "B_ai_prob_ewma",
    "B_ai_acc_ewma",
    "B_ai_err_rolling20",
    "B_xai_proc_score_ewma",
    "B_purpose_align_ewma",
}


def validate_history_metadata(
    features,
    history_shifted,
    outcomes_available,
):
    """
    Verify declared provenance of historical features.

    history_shifted:
        True if all history features exclude the
        current decision and future observations.

    outcomes_available:
        True if historical outcome-derived values
        were available at prediction time.

    This validates metadata, NOT feature construction.
    """

    selected = set(features) & HISTORY_FEATURES

    if not selected:
        return True

    if not history_shifted:
        raise ValueError(
            "Historical features must exclude "
            "current and future observations."
        )

    outcome_features = selected & {
        "B_ai_acc_ewma",
        "B_ai_err_rolling20",
    }

    if outcome_features and not outcomes_available:
        raise ValueError(
            "Historical AI correctness features require "
            "past outcomes available at prediction time."
        )

    return True


# -------------------------------------------------------
# 4. Unified evaluation gate
# -------------------------------------------------------

def validate_evaluation(
    train_df,
    test_df,
    features,
    target="override",
    id_col="instance_id",
    group_col="user_id",
    require_group_separation=True,
    history_shifted=False,
    outcomes_available=False,
):
    """
    Run leakage checks before an evaluation.

    Only returns sanitized X/y references after
    all applicable checks pass.
    """

    required = set(features) | {target}

    for name, df in (
        ("train", train_df),
        ("test", test_df),
    ):
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name}: missing columns {sorted(missing)}"
            )

    X_train = train_df.loc[:, features].copy()
    X_test = test_df.loc[:, features].copy()

    validate_features(X_train, target)
    validate_features(X_test, target)

    validate_split(
        train_df,
        test_df,
        id_col=id_col,
        group_col=group_col,
        require_group_separation=require_group_separation,
    )

    validate_history_metadata(
        features,
        history_shifted=history_shifted,
        outcomes_available=outcomes_available,
    )

    y_train = train_df[target].copy()
    y_test = test_df[target].copy()

    if not y_train.isin([0, 1]).all():
        raise ValueError("Invalid training target values.")

    if not y_test.isin([0, 1]).all():
        raise ValueError("Invalid testing target values.")

    if y_train.nunique() < 2:
        raise ValueError(
            "Training dataset requires both override classes."
        )

    return X_train, X_test, y_train, y_test
