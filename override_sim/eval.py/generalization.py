"""
This module is to asses transferability of the synthetic dataset. 
a good simulation holds only if it's successful in other conditions/worlds. 
It's also helpful to avoid model_in-model_out or extracting the same insights we incripted
in the dataset then claim it's new discovered information.

Tests:
    1. Within-world:
       Train and test on separate users within the same world.

    2. Cross-world (OOD):
       Train on one world, test on another.

    3. Leave-one-world-out (LOWO):
       Train on two worlds, test on the remaining world.

Evaluation:
    - ROC-AUC
    - PR-AUC
    - Brier score
    - Accuracy
    - Confusion matrix

All tests use observable features only.
"""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit

from .metrics import compute_metrics


# -------------------------------------------------------
# Shared helpers
# -------------------------------------------------------

def _validate_worlds(worlds, features, target, group_col):
    """Check required columns and basic input validity."""

    if not worlds:
        raise ValueError("No worlds provided.")

    required = set(features) | {target, group_col}

    for name, df in worlds.items():
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name}: missing columns {sorted(missing)}"
            )

        if df.empty:
            raise ValueError(f"{name}: dataset is empty.")

        if df[list(features)].isna().any().any():
            raise ValueError(
                f"{name}: missing feature values. "
                "Use an imputation pipeline if needed."
            )

        if not df[target].isin([0, 1]).all():
            raise ValueError(
                f"{name}: target must contain only 0 and 1."
            )


def _evaluate(
    model,
    train_df,
    test_df,
    features,
    target,
    train_world,
    test_world,
    protocol,
    threshold=0.5,
):
    """Fit a fresh model and evaluate held-out predictions."""

    estimator = clone(model)

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    if y_train.nunique() < 2:
        raise ValueError(
            f"{train_world}: training data contains "
            "only one override class."
        )

    estimator.fit(X_train, y_train)

    classes = list(estimator.classes_)

    if 1 not in classes:
        raise ValueError("Model has no override=1 class.")

    positive_index = classes.index(1)

    y_proba = estimator.predict_proba(X_test)[
        :, positive_index
    ]

    metrics = compute_metrics(
        y_true=y_test,
        y_prob=y_proba,
        threshold=threshold,
    )

    return {
        "protocol": protocol,
        "train_world": train_world,
        "test_world": test_world,
        "n_train": len(train_df),
        "n_test": len(test_df),
        **metrics,
    }


# -------------------------------------------------------
# 1. Within-world generalization
# -------------------------------------------------------

def evaluate_within_world(
    model,
    worlds,
    features,
    target="override",
    group_col="user_id",
    test_size=0.25,
    random_state=42,
    threshold=0.5,
):
    """
    Train and test within each world using different users.

    User-grouped splitting prevents the same user's
    historical behavior from appearing in both sets.
    """

    _validate_worlds(worlds, features, target, group_col)

    results = []

    for world_name, df in worlds.items():

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )

        train_idx, test_idx = next(
            splitter.split(
                df[features],
                df[target],
                groups=df[group_col],
            )
        )

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        results.append(
            _evaluate(
                model,
                train_df,
                test_df,
                features,
                target,
                train_world=world_name,
                test_world=world_name,
                protocol="within_world",
                threshold=threshold,
            )
        )

    return pd.DataFrame(results)


# -------------------------------------------------------
# 2. Cross-world generalization (OOD)
# -------------------------------------------------------

def evaluate_cross_world(
    model,
    worlds,
    features,
    target="override",
    group_col="user_id",
    threshold=0.5,
):
    """
    Train on one world and test on every other world.

    Example:
        W0  -> W_B
        W0  -> W_U
        W_B -> W0
        W_B -> W_U
        W_U -> W0
        W_U -> W_B

    Different worlds are intentionally evaluated
    without mixing their observations.
    """

    _validate_worlds(worlds, features, target, group_col)

    results = []

    for train_name, train_df in worlds.items():
        for test_name, test_df in worlds.items():

            if train_name == test_name:
                continue

            results.append(
                _evaluate(
                    model,
                    train_df,
                    test_df,
                    features,
                    target,
                    train_world=train_name,
                    test_world=test_name,
                    protocol="cross_world",
                    threshold=threshold,
                )
            )

    return pd.DataFrame(results)


# -------------------------------------------------------
# 3. Leave-one-world-out generalization
# -------------------------------------------------------

def evaluate_leave_one_world_out(
    model,
    worlds,
    features,
    target="override",
    group_col="user_id",
    threshold=0.5,
):
    """
    Train on all worlds except one, then evaluate
    on the excluded world.

    Example:
        W_B + W_U -> W0
        W0 + W_U  -> W_B
        W0 + W_B  -> W_U
    """

    _validate_worlds(worlds, features, target, group_col)

    if len(worlds) < 3:
        raise ValueError(
            "Leave-one-world-out requires at least 3 worlds."
        )

    results = []

    for test_name, test_df in worlds.items():

        training_worlds = {
            name: df
            for name, df in worlds.items()
            if name != test_name
        }

        train_df = pd.concat(
            training_worlds.values(),
            ignore_index=True,
        )

        train_name = "+".join(training_worlds.keys())

        results.append(
            _evaluate(
                model,
                train_df,
                test_df,
                features,
                target,
                train_world=train_name,
                test_world=test_name,
                protocol="leave_one_world_out",
                threshold=threshold,
            )
        )

    return pd.DataFrame(results)


# -------------------------------------------------------
# 4. Unified runner
# -------------------------------------------------------

def evaluate_generalization(
    model,
    worlds,
    features,
    target="override",
    group_col="user_id",
    test_size=0.25,
    random_state=42,
    threshold=0.5,
):
    """
    Run all three generalization protocols.

    Returns:
        DataFrame containing one row per evaluation.
    """

    within = evaluate_within_world(
        model=model,
        worlds=worlds,
        features=features,
        target=target,
        group_col=group_col,
        test_size=test_size,
        random_state=random_state,
        threshold=threshold,
    )

    cross = evaluate_cross_world(
        model=model,
        worlds=worlds,
        features=features,
        target=target,
        group_col=group_col,
        threshold=threshold,
    )

    lowo = evaluate_leave_one_world_out(
        model=model,
        worlds=worlds,
        features=features,
        target=target,
        group_col=group_col,
        threshold=threshold,
    )

    return pd.concat(
        [within, cross, lowo],
        ignore_index=True,
    )
