"""from __future__ import annotations
import pandas as pd

from override_sim.config.presets import phase1
from override_sim.sim.generator import Simulator
from override_sim.data.schema import OBSERVABLE_COLS
from override_sim.features.blocks import FeatureBuilder
from override_sim.features.leakage import LeakageGuard
from override_sim.models.baselines import LogisticOverrideModel, GBDTOverrideModel
from override_sim.eval.metrics import compute_metrics


def time_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def main():
    cfg = phase1()
    sim = Simulator(cfg)
    df = sim.run(return_latents=True)
    print("\n[Sanity] Override rate by stake (quartiles):")
    print(df.assign(stake_q=pd.qcut(df["stake"], 4, labels=False))
        .groupby("stake_q")["override"].mean())

    print("\n[Sanity] Override rate by audit (0/1):")
    print(df.groupby("audit")["override"].mean())

    print("\n[Sanity] Override rate by AI uncertainty (quartiles):")
    unc = 1.0 - (df["ai_prob"] - 0.5).abs() * 2.0
    print(df.assign(uq=pd.qcut(unc, 4, labels=False))
        .groupby("uq")["override"].mean())

    print(f"[Sim] rows={len(df)}  override_rate={df['override'].mean():.3f}  (target={cfg.target_override_rate})")
    print(f"[Sim] intercept_b={df.attrs.get('cfg.intercept_b')}")

    # Observable view
    df_obs = df[OBSERVABLE_COLS + ["override","y_true"]].copy()

    # Feature build (uses shifted override history proxies, still OK: it reflects past behavior logs)
    fb = FeatureBuilder()
    X = fb.build(df_obs)
    y = df_obs["override"].astype(int).to_numpy()

    # Leakage check
    LeakageGuard().assert_no_forbidden_columns(X)

    # Time split
    df_train, df_val, df_test = time_split(df_obs)
    X_train = fb.build(df_train)
    y_train = df_train["override"].astype(int).to_numpy()
    X_val = fb.build(df_val)
    y_val = df_val["override"].astype(int).to_numpy()
    X_test = fb.build(df_test)
    y_test = df_test["override"].astype(int).to_numpy()

    feature_sets = {
    "B_only": [c for c in X_train.columns if c.startswith("B_")],
    "U_only": [c for c in X_train.columns if c.startswith("U_")],
    "I_only": [c for c in X_train.columns if c.startswith("I_")],
    "All": list(X_train.columns),
}

    for fs_name, cols in feature_sets.items():
        lr = LogisticOverrideModel().fit(X_train[cols], y_train)
        p_test = lr.predict_proba(X_test[cols])
        m = compute_metrics(y_test, p_test)
        print(f"\n[Ablation Logistic] {fs_name}: {m}")
        # Train models
        lr = LogisticOverrideModel().fit(X_train, y_train)
        gb = GBDTOverrideModel().fit(X_train, y_train)

    # Evaluate
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
"""
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
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def print_sanity(df: pd.DataFrame) -> None:
    print("\n[Sanity] Override rate by stake (quartiles):")
    print(
        df.assign(stake_q=pd.qcut(df["stake"], 4, labels=False))
        .groupby("stake_q")["override"]
        .mean()
    )

    print("\n[Sanity] Override rate by audit (0/1):")
    print(df.groupby("audit")["override"].mean())

    print("\n[Sanity] Override rate by AI uncertainty (quartiles):")
    unc = 1.0 - (df["ai_prob"] - 0.5).abs() * 2.0
    print(
        df.assign(uq=pd.qcut(unc, 4, labels=False))
        .groupby("uq")["override"]
        .mean()
    )
    print("\n[Sanity] Override rate by policy_strict (quartiles):")
    print(df.assign(pq=pd.qcut(df["policy_strict"], 4, labels=False)).groupby("pq")["override"].mean())
    print("\n[Sanity] Override rate by recent AI errors proxy (quartiles):")
    # uses latent trust proxy if present; otherwise you can compute a proxy from columns later
    if "trust_T_latent" in df.columns:
        print(df.assign(tq=pd.qcut(df["trust_T_latent"], 4, labels=False)).groupby("tq")["override"].mean())



def run_ablations(X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, y_test) -> None:
    feature_sets = {
        "B_only": [c for c in X_train.columns if c.startswith("B_")],
        "U_only": [c for c in X_train.columns if c.startswith("U_")],
        "I_only": [c for c in X_train.columns if c.startswith("I_")],
        "All": list(X_train.columns),
    }

    for fs_name, cols in feature_sets.items():
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
    # IMPORTANT:
    # - We include y_true ONLY so trust_proxy can compute past AI correctness with SHIFT (no peeking).
    # - y_true is never included as a feature column; LeakageGuard checks the feature matrix.
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

    # 5) Time split (simulate “train on past, test on future”)
    df_train, df_val, df_test = time_split(df_obs)

    X_train = fb.build(df_train)
    y_train = df_train["override"].astype(int).to_numpy()

    X_val = fb.build(df_val)
    y_val = df_val["override"].astype(int).to_numpy()

    X_test = fb.build(df_test)
    y_test = df_test["override"].astype(int).to_numpy()

    # 6) Ablations
    run_ablations(X_train, y_train, X_test, y_test)

    # 7) Train baselines
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
