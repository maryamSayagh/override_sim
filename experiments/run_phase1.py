from __future__ import annotations

import pandas as pd

from override_sim.config.presets import phase1
from override_sim.sim.generator import Simulator
from override_sim.data.schema import OBSERVABLE_COLS
from override_sim.features.blocks import FeatureBuilder
from override_sim.features.leakage import LeakageGuard
from override_sim.models.baselines import LogisticOverrideModel, GBDTOverrideModel
from override_sim.eval.metrics import compute_metrics


def time_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """Simple chronological split: train (past) / val / test (future)."""
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def print_sanity(df: pd.DataFrame) -> None:
    print("\n[Sanity] Override rate by stake (quartiles):")
    print(
        df.assign(stake_q=pd.qcut(df["stake"], 4, labels=False, duplicates="drop"))
        .groupby("stake_q")["override"]
        .mean()
    )

    print("\n[Sanity] Override rate by audit (0/1):")
    print(df.groupby("audit")["override"].mean())

    print("\n[Sanity] Override rate by AI uncertainty (quartiles):")
    unc = 1.0 - (df["ai_prob"] - 0.5).abs() * 2.0
    print(
        df.assign(uq=pd.qcut(unc, 4, labels=False, duplicates="drop"))
        .groupby("uq")["override"]
        .mean()
    )

    if "policy_strict" in df.columns:
        print("\n[Sanity] Override rate by policy_strict (quartiles):")
        print(
            df.assign(pq=pd.qcut(df["policy_strict"], 4, labels=False, duplicates="drop"))
            .groupby("pq")["override"]
            .mean()
        )

    # Latent trust sanity (label it correctly!)
    if "trust_T_latent" in df.columns:
        print("\n[Sanity] Override rate by latent trust T (quartiles):")
        print(
            df.assign(tq=pd.qcut(df["trust_T_latent"], 4, labels=False, duplicates="drop"))
            .groupby("tq")["override"]
            .mean()
        )

    # Recent AI errors proxy sanity (the actual thing you intended)
    print("\n[Sanity] Override rate by recent AI errors proxy (quartiles):")
    if "B_ai_err_rolling20" in df.columns:
        print(
            df.assign(eq=pd.qcut(df["B_ai_err_rolling20"], 4, labels=False, duplicates="drop"))
            .groupby("eq")["override"]
            .mean()
        )
    elif "y_true" in df.columns and "ai_rec" in df.columns:
        # fallback: compute a simple rolling error proxy (not grouped)
        ai_correct = (df["ai_rec"] == df["y_true"]).astype(float)
        ai_err_proxy = (1.0 - ai_correct).rolling(20, min_periods=1).mean()
        print(
            df.assign(eq=pd.qcut(ai_err_proxy, 4, labels=False, duplicates="drop"))
            .groupby("eq")["override"]
            .mean()
        )
    else:
        print("  (skipped: no B_ai_err_rolling20 and no y_true/ai_rec to compute a proxy)")


def run_ablations(X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, y_test) -> None:
    feature_sets = {
        "B_only": [c for c in X_train.columns if c.startswith("B_")],
        "U_only": [c for c in X_train.columns if c.startswith("U_")],
        "I_only": [c for c in X_train.columns if c.startswith("I_")],
        "All": list(X_train.columns),
    }

    for fs_name, cols in feature_sets.items():
        if not cols:
            print(f"\n[Ablation Logistic] {fs_name}: (skipped: empty feature set)")
            continue

        lr = LogisticOverrideModel().fit(X_train[cols], y_train)
        p_test = lr.predict_proba(X_test[cols])
        m = compute_metrics(y_test, p_test)
        print(f"\n[Ablation Logistic] {fs_name}: {m}")


def main():
    # 1) Simulate
    cfg = phase1()
    sim = Simulator(cfg)
    df = sim.run(return_latents=True)

    print(f"[Sim] rows={len(df)}  override_rate={df['override'].mean():.3f}  (target={cfg.target_override_rate})")
    print(f"[Sim] intercept_b={df.attrs.get('cfg.intercept_b')}")
    print_sanity(df)

    # 2) Observable view for feature building
    # Include y_true ONLY for building lagged trust proxies inside FeatureBuilder.
    # LeakageGuard ensures y_true is not used as a feature column.
    df_obs = df[OBSERVABLE_COLS + ["override", "y_true"]].copy()

    # 3) Feature build
    fb = FeatureBuilder()
    X_all = fb.build(df_obs)
    y_all = df_obs["override"].astype(int).to_numpy()

    print("\n[Debug] Feature columns:", list(X_all.columns))
    print("[Debug] Has B_ai_acc_ewma?", "B_ai_acc_ewma" in X_all.columns)
    print("[Debug] Has B_ai_err_rolling20?", "B_ai_err_rolling20" in X_all.columns)
    print("[Debug] Has I_audit_x_stake?", "I_audit_x_stake" in X_all.columns)
    print("[Debug] Has I_audit_x_uncertainty?", "I_audit_x_uncertainty" in X_all.columns)

    # 4) Leakage check (feature matrix must not contain forbidden columns)
    LeakageGuard().assert_no_forbidden_columns(X_all)

    # 5) Time split (train on past, test on future)
    df_train, df_val, df_test = time_split(df_obs)

    X_train = fb.build(df_train)
    y_train = df_train["override"].astype(int).to_numpy()

    X_val = fb.build(df_val)
    y_val = df_val["override"].astype(int).to_numpy()

    X_test = fb.build(df_test)
    y_test = df_test["override"].astype(int).to_numpy()

    # 6) Ablations
    run_ablations(X_train, y_train, X_test, y_test)

    # 7) Train baselines (full feature set)
    lr = LogisticOverrideModel().fit(X_train, y_train)
    gb = GBDTOverrideModel().fit(X_train, y_train)

    # 8) Evaluate
    for name, model in [("Logistic", lr), ("GBDT", gb)]:
        p_val = model.predict_proba(X_val)
        p_test = model.predict_proba(X_test)

        m_val = compute_metrics(y_val, p_val)
        m_test = compute_metrics(y_test, p_test)

        print(f"\n== {name} ==")
        print("[VAL ]", m_val)
        print("[TEST]", m_test)


if __name__ == "__main__":
    main()
