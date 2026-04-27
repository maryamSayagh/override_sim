"""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from override_sim.data.schema import ALLOWED_FEATURE_COLS
from override_sim.features.blocks import FeatureBuilder
from override_sim.features.leakage import LeakageGuard
from override_sim.models.baselines import GBDTOverrideModel , RandomForestOverrideModel
from override_sim.eval.metrics import compute_metrics

DATA_DIR = Path("override_sim/hover/data")
OUT_DIR = Path("override_sim/hover/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "rf"   # options: "gbdt", "rf" 
WORLD_FILES = {
    "W0_baseline": DATA_DIR / "W0_baseline.csv",
    "W_U_dominant": DATA_DIR / "W_U_dominant.csv",
    "W_B_dominant": DATA_DIR / "W_B_dominant.csv",
}

TRAIN_MODE = "single"   # options: "single", "combined"

TRAIN_WORLD = "W0_baseline"
CROSS_WORLD = "W_U_dominant"

COMBINED_TRAIN_WORLDS = ["W0_baseline", "W_B_dominant", "W_U_dominant"]
EVAL_WORLDS = ["W0_baseline", "W_B_dominant", "W_U_dominant"]


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


def save_json(path: Path, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_worlds():
    dfs = {}
    for name, path in WORLD_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        dfs[name] = pd.read_csv(path)
    return dfs

def build_model(model_name: str):
    if model_name == "gbdt":
        return GBDTOverrideModel(
            max_depth=2,
            n_estimators=80,
            learning_rate=0.08,
        )
    elif model_name == "rf":
        return RandomForestOverrideModel(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    

def combine_worlds(dfs: dict[str, pd.DataFrame], world_names: list[str]) -> pd.DataFrame:
    parts = []
    for name in world_names:
        df = dfs[name].copy()
        df["source_world"] = name
        parts.append(df)
    return pd.concat(parts, axis=0, ignore_index=True)

def main():
    fb = FeatureBuilder()
    dfs = load_worlds()

    # Build all X/y once
    Xy = {}
    for world_name, df in dfs.items():
        X, y = build_Xy(df, fb)
        Xy[world_name] = {"X": X, "y": y}

    world_pairs = [
        ("W0_baseline", "W0_baseline"),
        ("W0_baseline", "W_U_dominant"),
        ("W_U_dominant", "W_U_dominant"),
        ("W_U_dominant", "W0_baseline"),
        ("W_B_dominant", "W_B_dominant"),
        ("W_B_dominant", "W0_baseline"),
        ("W_B_dominant", "W_U_dominant"),
        ("W0_baseline", "W_B_dominant")
    
    ]

    results = {}
    rows = []

    for train_world, test_world in world_pairs:
        X_train = Xy[train_world]["X"]
        y_train = Xy[train_world]["y"]

        X_test_raw = Xy[test_world]["X"]
        y_test = Xy[test_world]["y"]
        X_test = align_columns(X_train, X_test_raw)

        model = build_model('rf').fit(X_train, y_train)

        p_train = model.predict_proba(X_train)
        p_test = model.predict_proba(X_test)

        train_metrics = compute_metrics(y_train, p_train)
        test_metrics = compute_metrics(y_test, p_test)

        exp_name = f"{train_world}__to__{test_world}"
        results[exp_name] = {
            "train_world": train_world,
            "test_world": test_world,
            "n_features": int(X_train.shape[1]),
            "feature_names": list(X_train.columns),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "model_name": MODEL_NAME
    
        }

        rows.append({
            "train_world": train_world,
            "test_world": test_world,
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
            "brier": test_metrics["brier"],
            "pred_mean": test_metrics["pred_mean"],
            "label_mean": test_metrics["label_mean"],
        })

        pd.DataFrame({
            "y_true": y_test,
            "p_override_pred": p_test,
        }).to_csv(OUT_DIR / f"{exp_name}_predictions.csv", index=False)
        
        print(f"Model: {MODEL_NAME}")
        print(f"\n=== {train_world} -> {test_world} ===")
        print("Train metrics:")
        print(train_metrics)
        print("Test metrics:")
        print(test_metrics)

    summary = pd.DataFrame(rows)
    print("\n=== 2x2 HOVER summary ===")
    print(summary.round(4).to_string(index=False))

    summary.to_csv(OUT_DIR / "hover_2x2_summary.csv", index=False)
    save_json(OUT_DIR / "hover_2x2_metrics.json", results)

    print("\nSaved:")
    print(OUT_DIR / "hover_2x2_summary.csv")
    print(OUT_DIR / "hover_2x2_metrics.json")


if __name__ == "__main__":
    main()
    """

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from override_sim.data.schema import ALLOWED_FEATURE_COLS
from override_sim.features.blocks import FeatureBuilder
from override_sim.features.leakage import LeakageGuard
from override_sim.models.baselines import GBDTOverrideModel, RandomForestOverrideModel
from override_sim.eval.metrics import compute_metrics

DATA_DIR = Path("override_sim/hover/data")
OUT_DIR = Path("override_sim/hover/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gbdt"   # options: "gbdt", "rf"
WORLD_FILES = {
    "W0_baseline": DATA_DIR / "W0_baseline.csv",
    "W_U_dominant": DATA_DIR / "W_U_dominant.csv",
    "W_B_dominant": DATA_DIR / "W_B_dominant.csv",
}

TRAIN_MODE = "single"   # options: "single", "combined"
TRAIN_WORLD = "W0_baseline"
CROSS_WORLD = "W_U_dominant"
COMBINED_TRAIN_WORLDS = ["W0_baseline", "W_B_dominant", "W_U_dominant"]
EVAL_WORLDS = ["W0_baseline", "W_B_dominant", "W_U_dominant"]

VAL_SIZE = 0.15       # fraction of train data held out for validation
VAL_RANDOM_STATE = 42


def build_Xy(df: pd.DataFrame, fb: FeatureBuilder):
    if "override" not in df.columns:
        raise ValueError("Missing target column: override")

    X = fb.build(df)
    keep_cols = [c for c in X.columns if c in ALLOWED_FEATURE_COLS]
    X = X[keep_cols].copy()

    LeakageGuard().assert_no_forbidden_columns(X)

    y = df["override"].astype(int).to_numpy()
    return X, y


def split_train_val(
    X: pd.DataFrame,
    y: np.ndarray,
    val_size: float = VAL_SIZE,
    random_state: int = VAL_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Stratified train/val split, preserving class balance."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_val, y_train, y_val

def split_train_val_test_by_user(
    df: pd.DataFrame,
    fb: FeatureBuilder,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = VAL_RANDOM_STATE,
):
    """
    Split users into train / val / test groups FIRST, then build features
    separately on each partition so history never crosses boundaries.
    """
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

    return X_train, X_val, X_test, y_train, y_val, y_test

def align_columns(X_ref: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    return X_other.reindex(columns=X_ref.columns, fill_value=0.0)


def save_json(path: Path, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_worlds():
    dfs = {}
    for name, path in WORLD_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        dfs[name] = pd.read_csv(path)
    return dfs


def build_model(model_name: str):
    if model_name == "gbdt":
        return GBDTOverrideModel(
            max_depth=2,
            n_estimators=80,
            learning_rate=0.08,
        )
    elif model_name == "rf":
        return RandomForestOverrideModel(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def combine_worlds(dfs: dict[str, pd.DataFrame], world_names: list[str]) -> pd.DataFrame:
    parts = []
    for name in world_names:
        df = dfs[name].copy()
        df["source_world"] = name
        parts.append(df)
    return pd.concat(parts, axis=0, ignore_index=True)


def main():
    fb = FeatureBuilder()
    dfs = load_worlds()

    # Build all X/y once
    Xy = {}
    for world_name, df in dfs.items():
        X, y = build_Xy(df, fb)
        Xy[world_name] = {"X": X, "y": y}

    world_pairs = [
        ("W0_baseline",   "W0_baseline"),
        ("W0_baseline",   "W_U_dominant"),
        ("W_U_dominant",  "W_U_dominant"),
        ("W_U_dominant",  "W0_baseline"),
        ("W_B_dominant",  "W_B_dominant"),
        ("W_B_dominant",  "W0_baseline"),
        ("W_B_dominant",  "W_U_dominant"),
        ("W0_baseline",   "W_B_dominant"),
    ]

    results = {}
    rows = []

    for train_world, test_world in world_pairs:
        X_train, X_val, X_test_same, y_train, y_val, y_test_same = split_train_val_test_by_user(
    dfs[train_world], fb
)
        # --- Validation split (stratified, from training world only) ---
        if train_world == test_world:
            X_test = align_columns(X_train, X_test_same)
            y_test = y_test_same
        else:
            X_test_raw = Xy[test_world]["X"]
            y_test = Xy[test_world]["y"]
            X_test = align_columns(X_train, X_test_raw)
        # --- Fit on train split only ---
        model = build_model(MODEL_NAME).fit(X_train, y_train)

        p_train = model.predict_proba(X_train)
        p_val   = model.predict_proba(X_val)
        p_test  = model.predict_proba(X_test)

        train_metrics = compute_metrics(y_train, p_train)
        val_metrics   = compute_metrics(y_val,   p_val)
        test_metrics  = compute_metrics(y_test,  p_test)

        exp_name = f"{train_world}__to__{test_world}"
        results[exp_name] = {
            "train_world":   train_world,
            "test_world":    test_world,
            "n_train":       int(len(y_train)),
            "n_val":         int(len(y_val)),
            "n_features":    int(X_train.shape[1]),
            "feature_names": list(X_train.columns),
            "train_metrics": train_metrics,
            "val_metrics":   val_metrics,
            "test_metrics":  test_metrics,
            "model_name":    MODEL_NAME,
        }

        rows.append({
            "train_world":    train_world,
            "test_world":     test_world,
            # val columns come first so overfitting is visible at a glance
            "val_roc_auc":    val_metrics["roc_auc"],
            "val_pr_auc":     val_metrics["pr_auc"],
            "val_brier":      val_metrics["brier"],
            "roc_auc":        test_metrics["roc_auc"],
            "pr_auc":         test_metrics["pr_auc"],
            "brier":          test_metrics["brier"],
            "pred_mean":      test_metrics["pred_mean"],
            "label_mean":     test_metrics["label_mean"],
        })

        # Save per-experiment predictions (val + test)
        pd.DataFrame({
            "y_true":          y_val,
            "p_override_pred": p_val,
            "split":           "val",
        }).to_csv(OUT_DIR / f"{exp_name}_val_predictions.csv", index=False)

        pd.DataFrame({
            "y_true":          y_test,
            "p_override_pred": p_test,
            "split":           "test",
        }).to_csv(OUT_DIR / f"{exp_name}_predictions.csv", index=False)

        print(f"Model: {MODEL_NAME}")
        print(f"\n=== {train_world} -> {test_world} ===")
        print(f"  Train n={len(y_train)}, Val n={len(y_val)}, Test n={len(y_test)}")
        print("Train metrics:"); print(train_metrics)
        print("Val   metrics:"); print(val_metrics)
        print("Test  metrics:"); print(test_metrics)

    summary = pd.DataFrame(rows)
    print("\n=== 2x2 HOVER summary ===")
    print(summary.round(4).to_string(index=False))

    summary.to_csv(OUT_DIR / "hover_2x2_summary.csv", index=False)
    save_json(OUT_DIR / "hover_2x2_metrics.json", results)

    print("\nSaved:")
    print(OUT_DIR / "hover_2x2_summary.csv")
    print(OUT_DIR / "hover_2x2_metrics.json")


if __name__ == "__main__":
    main()