from __future__ import annotations

import pandas as pd


def _ewma_shifted(s: pd.Series, alpha: float, shift: int) -> pd.Series:
    return s.shift(shift).ewm(alpha=alpha, adjust=False).mean()


def add_behavior_history_proxies(
    df: pd.DataFrame,
    group_col: str = "user_id",
) -> pd.DataFrame:
    """
    Behavioral trust proxies from past override behavior only.
    These are always observable from logged user actions.
    """
    out = df.copy()

    out["B_ovr_ewma"] = (
        out.groupby(group_col)["override"]
        .transform(lambda s: _ewma_shifted(s, alpha=0.05, shift=1))
        .fillna(0.35)
    )

    out["B_ovr_rolling10"] = (
        out.groupby(group_col)["override"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        .fillna(0.35)
    )

    return out


def add_ai_history_proxies(
    df: pd.DataFrame,
    group_col: str = "user_id",
    outcome_lag: int = 0,
) -> pd.DataFrame:
    """
    Proxies based on past AI outputs and past observed AI correctness.

    - AI probability history is always observable
    - AI correctness history is only observable after the outcome is available
    """
    out = df.copy()

    # AI confidence history (always available)
    out["B_ai_prob_ewma"] = (
        out.groupby(group_col)["ai_prob"]
        .transform(lambda s: _ewma_shifted(s, alpha=0.05, shift=1))
        .fillna(0.5)
    )

    # Past AI correctness (only if y_true is present in the dataframe)
    if "y_true" in out.columns:
        ai_correct = (out["ai_rec"] == out["y_true"]).astype(float)

        out["B_ai_acc_ewma"] = (
            ai_correct.groupby(out[group_col])
            .transform(lambda s: _ewma_shifted(s, alpha=0.05, shift=1 + outcome_lag))
            .fillna(0.74)
        )

        out["B_ai_err_rolling20"] = (
            (1.0 - ai_correct)
            .groupby(out[group_col])
            .transform(lambda s: s.shift(1 + outcome_lag).rolling(20, min_periods=1).mean())
            .fillna(0.26)
        )
    else:
        out["B_ai_acc_ewma"] = 0.74
        out["B_ai_err_rolling20"] = 0.26

    return out


def add_process_proxies(
    df: pd.DataFrame,
    group_col: str = "user_id",
) -> pd.DataFrame:
    """
    Process proxies from explainability metadata.
    These are event-level observables, not latent trust itself.
    """
    out = df.copy()

    # Raw explainability indicators
    out["B_xai_present"] = out["xai_present"].astype(float)
    out["B_xai_pre"] = (out["xai_timing"] == 1).astype(float)
    out["B_xai_post"] = (out["xai_timing"] == 2).astype(float)

    # Process score proxy (same aggregation used in the simulator)
    out["B_xai_proc_score"] = (
        out["xai_present"].astype(float)
        * (
            0.35 * out["xai_detail"].astype(float)
            + 0.35 * out["xai_specificity"].astype(float)
            + 0.30 * out["xai_consistency"].astype(float)
        )
    )

    out["B_xai_faithfulness"] = out["xai_faithfulness"].astype(float)

    # Optional history of process exposure
    out["B_xai_proc_score_ewma"] = (
        out.groupby(group_col)["B_xai_proc_score"]
        .transform(lambda s: _ewma_shifted(s, alpha=0.05, shift=1))
        .fillna(0.0)
    )

    return out


def add_purpose_proxies(
    df: pd.DataFrame,
    group_col: str = "user_id",
) -> pd.DataFrame:
    """
    Purpose proxies from the observable purpose alignment signal.

    Important:
    - purpose_align is the observable proxy
    - any underlying true purpose structure remains latent and forbidden
    """
    out = df.copy()

    out["B_purpose_align"] = out["purpose_align"].astype(float)

    out["B_purpose_align_ewma"] = (
        out.groupby(group_col)["purpose_align"]
        .transform(lambda s: _ewma_shifted(s, alpha=0.05, shift=1))
        .fillna(0.5)
    )

    return out


def add_trust_proxies(
    df: pd.DataFrame,
    group_col: str = "user_id",
    outcome_lag: int = 0,
) -> pd.DataFrame:
    """
    Unified trust proxy builder.

    Combines separate observable proxy families:
    - behavior history
    - AI history / correctness history
    - process / explainability
    - purpose alignment

    The goal is not to reconstruct latent trust directly,
    but to expose behavioral traces associated with it.
    """
    out = df.copy()

    out = add_behavior_history_proxies(out, group_col=group_col)
    out = add_ai_history_proxies(out, group_col=group_col, outcome_lag=outcome_lag)
    out = add_process_proxies(out, group_col=group_col)
    out = add_purpose_proxies(out, group_col=group_col)

    return out