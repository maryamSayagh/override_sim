from __future__ import annotations
import numpy as np
import pandas as pd

from .trust_proxy import add_trust_proxies


class FeatureBuilder:
    """
    Builds observable-only B/U/I feature blocks.

    Notes:
    - We may pass y_true in the dataframe ONLY so add_trust_proxies() can compute
      *shifted* past AI correctness proxies (no lookahead).
    - We never include y_true itself (or any latent) in the returned feature matrix.
    """

    def build(self, df_observed_with_label: pd.DataFrame) -> pd.DataFrame:
        # Adds shifted history proxies: override EWMA, ai_prob EWMA,
        # plus (if y_true exists) ai accuracy proxies with a shift.
        df = add_trust_proxies(df_observed_with_label, group_col="user_id", outcome_lag=0)

        # -----------------------
        # Belief proxies (B)
        # -----------------------
        # Uncertainty: 1 at 0.5, 0 at 0 or 1
        df["B_ai_uncertainty"] = 1.0 - np.abs(df["ai_prob"] - 0.5) * 2.0
        df["B_ai_margin"] = np.abs(df["ai_prob"] - 0.5)
        df["B_ai_rec"] = df["ai_rec"].astype(int)
        # Process/explainability features (observable)
        df["B_xai_present"] = df["xai_present"].astype(int)
        df["B_xai_pre"] = (df["xai_timing"] == 1).astype(int) 
        df["B_xai_post"] = (df["xai_timing"] == 2).astype(int)
        df["B_xai_proc_score"] = df["xai_present"] * (
        0.35 * df["xai_detail"] + 0.35 * df["xai_specificity"] + 0.30 * df["xai_consistency"]
        )
        df["B_xai_faithfulness"] = df["xai_faithfulness"]
        df["B_purpose_align"] = df["purpose_align"]
        # -----------------------
        # Utility/Risk proxies (U)
        # -----------------------
        df["U_stake"] = df["stake"]
        df["U_risk_pressure"] = df["stake"] * (1.0 + df["time_pressure"])

        # -----------------------
        # Incentive proxies (I)
        # -----------------------
        # ------------------ Incentive features (I- block) ------------------
        audit  = df["audit"].fillna(0.0).astype(float)
        policy = df["policy_strict"].fillna(0.0).astype(float)

        df["I_audit"]         = audit
        df["I_policy"]        = policy
        df["I_audit_policy"]  = audit * policy

# Interactions used elsewhere
        uncertainty = (0.5 - (df["ai_prob"].astype(float) - 0.5).abs())
        df["I_audit_x_stake"]      = audit * df["stake"].astype(float)
        df["I_audit_x_uncertainty"] = audit * uncertainty
       
        # -----------------------
        # Cross-block interactions
        # -----------------------
        df["BU_uncertainty_x_stake"] = df["B_ai_uncertainty"] * df["U_stake"]
        df["BI_uncertainty_x_audit"] = df["B_ai_uncertainty"] * df["I_audit"]

        # Some trust proxies depend on y_true being present in input df.
        # If absent, add_trust_proxies() should create defaults, but we guard anyway.
        for col, default in [
            ("B_ovr_ewma", 0.35),
            ("B_ai_prob_ewma", 0.5),
            ("B_ovr_rolling10", 0.35),
            ("B_ai_acc_ewma", 0.5),
            ("B_ai_err_rolling20", 0.5),
        ]:
            if col not in df.columns:
                df[col] = default

        feature_cols = [
            # B (instant)
            "B_ai_uncertainty",
            "B_ai_margin",
            "B_ai_rec",

            # B (history proxies)
            "B_ovr_ewma",
            "B_ai_prob_ewma",
            "B_ovr_rolling10",
            "B_ai_acc_ewma",
            "B_ai_err_rolling20",
            # B (explainability/process)
            "B_xai_present",
            "B_xai_pre",
            "B_xai_post",
            "B_xai_proc_score",
            "B_xai_faithfulness",
            "B_purpose_align",
            # U
            "U_stake",
            "U_risk_pressure",

            # I
            "I_audit",
            "I_policy",
            "I_audit_policy",
            "I_audit_x_stake",
            "I_audit_x_uncertainty",

            # Interactions
            "BU_uncertainty_x_stake",
            "BI_uncertainty_x_audit",
        ]

        return df[feature_cols].astype(float)
