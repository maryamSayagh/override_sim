from __future__ import annotations

from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd

from override_sim.data.schema import ALLOWED_FEATURE_COLS
from override_sim.features.blocks import FeatureBuilder
from override_sim.features.leakage import LeakageGuard
from override_sim.models.baselines import RandomForestOverrideModel
from override_sim.eval.metrics import compute_metrics


DATA_DIR = Path("override_sim/hover/data")
OUT_DIR = Path("override_sim/hover/output/export_wb_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORLD_NAME = "W_B_dominant"
WORLD_FILE = DATA_DIR / f"{WORLD_NAME}.csv"

VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42


def build_Xy(df: pd.DataFrame, fb: FeatureBuilder):
    if "override" not in df.columns:
        raise ValueError("Missing target column: override")

    X = fb.build(df)
    keep_cols = [c for c in X.columns if c in ALLOWED_FEATURE_COLS]
    X = X[keep_cols].copy()

    LeakageGuard().assert_no_forbidden_columns(X)

    y = df["override"].astype(int).to_numpy()
    return X, y


def align_columns(X_ref: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    return X_other.reindex(columns=X_ref.columns, fill_value=0.0)


def split_train_val_test_by_user(
    df: pd.DataFrame,
    fb: FeatureBuilder,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    rng = np.random.default_rng(random_state)
    users = df["user_id"].unique()
    rng.shuffle(users)

    n_users = len(users)
    n_test = max(1, int(n_users * test_size))
    n_val = max(1, int(n_users * val_size))

    test_users = set(users[:n_test])
    val_users = set(users[n_test:n_test + n_val])
    train_users = set(users[n_test + n_val:])

    df_train = df[df["user_id"].isin(train_users)].copy()
    df_val = df[df["user_id"].isin(val_users)].copy()
    df_test = df[df["user_id"].isin(test_users)].copy()

    X_train, y_train = build_Xy(df_train, fb)
    X_val, y_val = build_Xy(df_val, fb)
    X_test, y_test = build_Xy(df_test, fb)

    X_val = align_columns(X_train, X_val)
    X_test = align_columns(X_train, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_json(path: Path, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    if not WORLD_FILE.exists():
        raise FileNotFoundError(f"Missing file: {WORLD_FILE}")

    fb = FeatureBuilder()
    df = pd.read_csv(WORLD_FILE)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test_by_user(df, fb)

    model = RandomForestOverrideModel(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
    ).fit(X_train, y_train)

    p_train = model.predict_proba(X_train)
    p_val = model.predict_proba(X_val)
    p_test = model.predict_proba(X_test)

    train_metrics = compute_metrics(y_train, p_train)
    val_metrics = compute_metrics(y_val, p_val)
    test_metrics = compute_metrics(y_test, p_test)

    joblib.dump(model, OUT_DIR / "hover_model.pkl")
    save_json(OUT_DIR / "feature_columns.json", {"feature_columns": list(X_train.columns)})

    pd.DataFrame({
        "y_true": y_train,
        "p_override_pred": p_train,
        "split": "train",
    }).to_csv(OUT_DIR / "train_predictions.csv", index=False)

    pd.DataFrame({
        "y_true": y_val,
        "p_override_pred": p_val,
        "split": "val",
    }).to_csv(OUT_DIR / "validation_predictions.csv", index=False)

    pd.DataFrame({
        "y_true": y_test,
        "p_override_pred": p_test,
        "split": "test",
    }).to_csv(OUT_DIR / "test_predictions.csv", index=False)

    metrics = {
        "world": WORLD_NAME,
        "model_name": "rf",
        "val_size": VAL_SIZE,
        "test_size": TEST_SIZE,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(OUT_DIR / "metrics.json", metrics)

    print("\n=== Exported HOVER model ===")
    print(f"World: {WORLD_NAME}")
    print(f"Train n={len(y_train)}, Val n={len(y_val)}, Test n={len(y_test)}")
    print("\nTrain metrics:")
    print(train_metrics)
    print("\nValidation metrics:")
    print(val_metrics)
    print("\nTest metrics:")
    print(test_metrics)

    print("\nSaved:")
    print(OUT_DIR / "hover_model.pkl")
    print(OUT_DIR / "feature_columns.json")
    print(OUT_DIR / "train_predictions.csv")
    print(OUT_DIR / "validation_predictions.csv")
    print(OUT_DIR / "test_predictions.csv")
    print(OUT_DIR / "metrics.json")


if __name__ == "__main__":
    main()
    