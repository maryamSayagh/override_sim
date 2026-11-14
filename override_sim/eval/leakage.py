

"""
 Leakage and Feature Access Validation

Checks:
    - Feature access permissions
    - Train/test instance and user overlap
    - Historical feature availability
    - Target validity

Failed checks raise exceptions with suggested fixes.
"""

import pandas as pd

from data.data_schema import (
    ALLOWED_FEATURE_COLS,
    FORBIDDEN_COLS,
    TARGET_COL,
)


# -------------------------------------------------------
# 1. Feature access validation
# -------------------------------------------------------

def validate_features(X):
    """Ensure model inputs contain only allowed features."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X must be a pandas DataFrame. "
            "Fix: Check feature construction in the experiment runner."
        )

    if X.empty:
        raise ValueError(
            "Feature dataset is empty. "
            "Fix: Check dataset loading, filtering, and feature selection."
        )

    if not X.columns.is_unique:
        raise ValueError(
            "Duplicate feature names detected. "
            "Fix: Check dataset merges and feature construction."
        )

    selected = set(X.columns)
    forbidden = selected & set(FORBIDDEN_COLS + [TARGET_COL])
    unknown = selected - set(ALLOWED_FEATURE_COLS)

    if forbidden:
        raise ValueError(
            f"Forbidden features detected: {sorted(forbidden)}. "
            "Fix: Remove them from the training feature list. "
            "Check data/data_schema.py and the dataset masking step."
        )

    if unknown:
        raise ValueError(
            f"Unregistered features detected: {sorted(unknown)}. "
            "Fix: Review these columns and register legitimate "
            "observable features in data/data_schema.py."
        )

    if X.isna().any().any():
        raise ValueError(
            "Missing feature values detected. "
            "Fix: Check data preparation and trust_proxy.py. "
            "Fit imputation only on the training partition."
        )

    return True


# -------------------------------------------------------
# 2. Train/test separation
# -------------------------------------------------------

def validate_split(
    train_df,
    test_df,
    id_row=("t", "user_id"),
    group_col="user_id",
    require_group_separation=True,
):
    """
    Validate training/testing separation.

    Checks:
        1. Required identifier columns exist.
        2. Identifiers contain no null values.
        3. (t, user_id) combinations are unique within each dataset.
        4. No decision appears in both training and testing.
        5. No users overlap when group separation is required.

    Set id_row=None for cross-world testing when
    identifiers are only unique within each world/run.
    """

    # -------------------------------------------------------
    # 1. Decision identifier validation
    # -------------------------------------------------------

    if id_row is not None:
        id_cols = list(id_row)

        for name, df in (
            ("train", train_df),
            ("test", test_df),
        ):

            # Check required columns
            missing = set(id_cols) - set(df.columns)

            if missing:
                raise ValueError(
                    f"{name}: missing ID columns {sorted(missing)}. "
                    "Fix: Check generator.py and dataset export."
                )

            # Check null identifiers
            if df[id_cols].isna().any().any():
                raise ValueError(
                    f"{name}: null decision identifiers detected. "
                    "Fix: Check time-step and user ID generation "
                    "in generator.py."
                )

            # Check duplicate decisions
            if df.duplicated(subset=id_cols).any():
                raise ValueError(
                    f"{name}: duplicate {tuple(id_cols)} "
                    "combinations detected. "
                    "Fix: Check generator.py to ensure each user "
                    "has only one decision per time step."
                )

        # ---------------------------------------------------
        # 2. Train/test decision overlap
        # ---------------------------------------------------

        train_ids = set(
            train_df[id_cols].itertuples(
                index=False,
                name=None,
            )
        )

        test_ids = set(
            test_df[id_cols].itertuples(
                index=False,
                name=None,
            )
        )

        overlap = train_ids & test_ids

        if overlap:
            raise ValueError(
                f"Train/test decision overlap: "
                f"{len(overlap)} matching identifiers. "
                "Fix: Check dataset splitting in the experiment "
                "runner. Ensure decisions are not reused "
                "between training and testing."
            )

    # -------------------------------------------------------
    # 3. User-group separation
    # -------------------------------------------------------

    if require_group_separation:

        for name, df in (
            ("train", train_df),
            ("test", test_df),
        ):

            if group_col not in df.columns:
                raise ValueError(
                    f"{name}: missing group column '{group_col}'. "
                    "Fix: Check dataset loading and preserve "
                    "user_id for grouped splitting."
                )

            if df[group_col].isna().any():
                raise ValueError(
                    f"{name}: null user IDs detected. "
                    "Fix: Check user assignment in generator.py."
                )

        train_users = set(train_df[group_col])
        test_users = set(test_df[group_col])

        user_overlap = train_users & test_users

        if user_overlap:
            raise ValueError(
                f"Train/test user overlap: "
                f"{len(user_overlap)} users. "
                "Fix: Use GroupShuffleSplit with user_id "
                "as the grouping variable in generalization.py."
            )

    return True

# -------------------------------------------------------
# 3. Historical feature checks
# -------------------------------------------------------

def validate_history_metadata(
    features,
    history_shifted,
    outcomes_available,
):
    """
    Validate declared historical feature availability.

    Uses the registered allowed features rather than
    maintaining a separate feature registry.

    This is a metadata check, not a timestamp audit.
    """

    history_features = {
        col for col in ALLOWED_FEATURE_COLS
        if col.endswith(("_ewma", "_rolling10", "_rolling20"))
    }

    selected = set(features) & history_features

    if not selected:
        return True

    if not history_shifted:
        raise ValueError(
            f"Historical features not verified: {sorted(selected)}. "
            "Fix: Check trust_proxy.py. Sort observations by user "
            "and chronological order before calculating shifted "
            "EWMA or rolling features."
        )

    outcome_features = {
        col for col in selected
        if col.startswith(("B_ai_acc_", "B_ai_err_"))
    }

    if outcome_features and not outcomes_available:
        raise ValueError(
            f"Outcome availability not verified: "
            f"{sorted(outcome_features)}. "
            "Fix: Check trust_proxy.py, outcome_lag, and "
            "when correctness outcomes become observable."
        )

    return True


# -------------------------------------------------------
# 4. Unified evaluation gate
# -------------------------------------------------------

def validate_evaluation(
    train_df,
    test_df,
    features,
    target=TARGET_COL,
    id_col=None,
    group_col="user_id",
    require_group_separation=True,
    history_shifted=False,
    outcomes_available=False,
):
    """
    Validate training and testing data before evaluation.

    Returns:
        X_train, X_test, y_train, y_test
    """

    if target != TARGET_COL:
        raise ValueError(
            f"Unexpected target '{target}'. "
            f"Expected '{TARGET_COL}'. "
            "Fix: Check TARGET_COL in data/data_schema.py."
        )

    if not features:
        raise ValueError(
            "No input features selected. "
            "Fix: Check feature selection in the experiment runner."
        )

    if len(features) != len(set(features)):
        raise ValueError(
            "Duplicate selected features. "
            "Fix: Remove repeated columns from the feature list."
        )

    required = set(features) | {target}

    for name, df in (
        ("train", train_df),
        ("test", test_df),
    ):
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name}: missing columns {sorted(missing)}. "
                "Fix: Check dataset loading, feature engineering, "
                "and data/data_schema.py."
            )

    X_train = train_df.loc[:, features].copy()
    X_test = test_df.loc[:, features].copy()

    validate_features(X_train)
    validate_features(X_test)

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
        raise ValueError(
            "Invalid training target values. "
            "Fix: Check binary override generation in the "
            "simulator and dataset preprocessing."
        )

    if not y_test.isin([0, 1]).all():
        raise ValueError(
            "Invalid testing target values. "
            "Fix: Check the test dataset's override column."
        )

    if y_train.nunique() < 2:
        raise ValueError(
            "Training dataset contains only one override class. "
            "Fix: Inspect override distribution and the "
            "train/test splitting procedure."
        )

    return X_train, X_test, y_train, y_test
