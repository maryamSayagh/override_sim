from __future__ import annotations
import pandas as pd


def add_trust_proxies(df: pd.DataFrame, group_col: str = "user_id", outcome_lag: int = 0) -> pd.DataFrame:
    """
    Observable-only trust proxies.
    - past override history (always observable)
    - past AI correctness history (observable once outcome is available; we simulate with outcome_lag)
    outcome_lag=0 means "outcome available immediately after event"
    outcome_lag=5 means "you only know correctness for events >=5 steps ago"
    """
    out = df.copy()

    def ewma_shifted(s: pd.Series, alpha: float, shift: int) -> pd.Series:
        return s.shift(shift).ewm(alpha=alpha, adjust=False).mean()

    # Past override behavior (always available)
    out["B_ovr_ewma"] = out.groupby(group_col)["override"].transform(lambda s: ewma_shifted(s, 0.05, 1)).fillna(0.35)
    out["B_ovr_rolling10"] = (
        out.groupby(group_col)["override"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        .fillna(0.35)
    )

    # AI confidence history (available)
    out["B_ai_prob_ewma"] = out.groupby(group_col)["ai_prob"].transform(lambda s: ewma_shifted(s, 0.05, 1)).fillna(0.5)

    # Past AI correctness (requires y_true to be present in df; only used with a lag to avoid peeking)
    if "y_true" in out.columns:
        ai_correct = (out["ai_rec"] == out["y_true"]).astype(float)
        out["B_ai_acc_ewma"] = ai_correct.groupby(out[group_col]).transform(
            lambda s: ewma_shifted(s, alpha=0.05, shift=1 + outcome_lag)
        ).fillna(0.5)
        out["B_ai_err_rolling20"] = (1.0 - ai_correct).groupby(out[group_col]).transform(
            lambda s: s.shift(1 + outcome_lag).rolling(20, min_periods=1).mean()
        ).fillna(0.5)
    else:
        out["B_ai_acc_ewma"] = 0.5
        out["B_ai_err_rolling20"] = 0.5

    return out
