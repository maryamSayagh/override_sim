# experiments/run_phase2.py
from __future__ import annotations

import pandas as pd

from override_sim.sim.generator import Simulator
from override_sim.data.schema import ALLOWED_FEATURE_COLS
from override_sim.features.blocks import FeatureBuilder
from override_sim.features.leakage import LeakageGuard
from override_sim.models.baselines import LogisticOverrideModel
from override_sim.eval.metrics import compute_metrics
from override_sim.config.schemas import SimConfig
from override_sim.config.worlds import build_all_world_cfgs, get_world_description


def time_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def build_Xy(df_obs: pd.DataFrame, fb: FeatureBuilder):
    X = fb.build(df_obs)

    # Keep only columns explicitly allowed for prediction, if present
    keep_cols = [c for c in X.columns if c in ALLOWED_FEATURE_COLS]
    X = X[keep_cols].copy()

    y = df_obs["override"].astype(int).to_numpy()
    return X, y


def align_columns(X_ref: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    """Reindex other to ref columns (safe for cross-world eval)."""
    return X_other.reindex(columns=X_ref.columns, fill_value=0.0)


def run_ablations(X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, y_test):
    """
    Logistic ablations on TEST.
    Returns dict: block -> metrics dict.
    """
    feature_sets = {
        "B_only": [c for c in X_train.columns if c.startswith("B_")],
        "U_only": [c for c in X_train.columns if c.startswith("U_")],
        "I_only": [c for c in X_train.columns if c.startswith("I_")],
        "All": list(X_train.columns),
    }

    out = {}
    for name, cols in feature_sets.items():
        if not cols:
            out[name] = {"roc_auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan")}
            continue

        lr = LogisticOverrideModel().fit(X_train[cols], y_train)
        p = lr.predict_proba(X_test[cols])
        out[name] = compute_metrics(y_test, p)

    return out


def main():
    # nicer printing
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 200)

    base_cfg = SimConfig(target_override_rate=0.25)
    worlds = build_all_world_cfgs(base_cfg)
    fb = FeatureBuilder()

    cache = {}

    print("\n=== Phase 2: Worlds ===")
    for wname, cfg in worlds.items():
        note = get_world_description(wname)

        sim = Simulator(cfg)
        df = sim.run(return_latents=True)

        # Keep the columns needed to build features and labels
        # ALLOWED_FEATURE_COLS should contain raw observable + derived allowed features,
        # but FeatureBuilder is expected to construct derived features from observables.
        # So here we keep the raw columns if present, plus override / y_true for feature building.
        raw_cols = [c for c in df.columns if c in ALLOWED_FEATURE_COLS]
        extra_cols = [c for c in ["override", "y_true"] if c in df.columns]
        df_obs = df[raw_cols + extra_cols].copy()

        df_train, df_val, df_test = time_split(df_obs)

        X_train, y_train = build_Xy(df_train, fb)
        X_val, y_val = build_Xy(df_val, fb)
        X_test, y_test = build_Xy(df_test, fb)

        LeakageGuard().assert_no_forbidden_columns(X_train)

        # within-world baseline (All features)
        model = LogisticOverrideModel().fit(X_train, y_train)
        m_test = compute_metrics(y_test, model.predict_proba(X_test))

        # ablations
        abl = run_ablations(X_train, y_train, X_test, y_test)

        cache[wname] = {
            "note": note,
            "override_rate": float(df["override"].mean()),
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "within_test": m_test,
            "ablations_test": abl,
        }

        print(
            f"\n[{wname}] {note}\n"
            f"  override_rate={cache[wname]['override_rate']:.3f}\n"
            f"  within-world TEST (All): {m_test}"
        )
        print("  ablations TEST (Logistic):")
        for k in ["B_only", "U_only", "I_only", "All"]:
            print(f"    - {k}: {abl[k]}")

    # ---- Cross-world matrix (train on i, test on j) ----
    print("\n=== Cross-world generalization (Logistic, All features) ===")
    names = list(cache.keys())

    roc = pd.DataFrame(index=names, columns=names, dtype=float)
    pr = pd.DataFrame(index=names, columns=names, dtype=float)
    brier = pd.DataFrame(index=names, columns=names, dtype=float)

    for train_name in names:
        Xtr = cache[train_name]["X_train"]
        ytr = cache[train_name]["y_train"]
        model = LogisticOverrideModel().fit(Xtr, ytr)

        for test_name in names:
            Xte_raw = cache[test_name]["X_test"]
            yte = cache[test_name]["y_test"]
            Xte = align_columns(Xtr, Xte_raw)

            p = model.predict_proba(Xte)
            m = compute_metrics(yte, p)

            roc.loc[train_name, test_name] = m["roc_auc"]
            pr.loc[train_name, test_name] = m["pr_auc"]
            brier.loc[train_name, test_name] = m["brier"]

    print("\nROC-AUC matrix:")
    print(roc.round(3).to_string())
    print("\nPR-AUC matrix:")
    print(pr.round(3).to_string())
    print("\nBrier matrix:")
    print(brier.round(3).to_string())

    # ---- Build ablation table ----
    ablation_rows = []
    for wname in names:
        abl = cache[wname]["ablations_test"]
        for block in ["B_only", "U_only", "I_only", "All"]:
            ablation_rows.append(
                {
                    "world": wname,
                    "block": block,
                    "roc_auc": abl[block]["roc_auc"],
                    "pr_auc": abl[block]["pr_auc"],
                    "brier": abl[block]["brier"],
                }
            )
    ablations = pd.DataFrame(ablation_rows).sort_values(["world", "block"])

    print("\n=== Ablations (TEST, Logistic) ===")
    print("\nAblation ROC-AUC:")
    print(ablations.pivot(index="world", columns="block", values="roc_auc").round(3).to_string())
    print("\nAblation PR-AUC:")
    print(ablations.pivot(index="world", columns="block", values="pr_auc").round(3).to_string())

    # ---- Save artifacts ----
    roc.to_csv("phase2_roc_auc.csv")
    pr.to_csv("phase2_pr_auc.csv")
    brier.to_csv("phase2_brier.csv")
    ablations.to_csv("phase2_ablations.csv", index=False)

    print("\n[Saved] phase2_roc_auc.csv, phase2_pr_auc.csv, phase2_brier.csv, phase2_ablations.csv")


if __name__ == "__main__":
    main()