# override_sim/sim/trust_channels.py
# -----------------------------------------------------------
# Lee & See evidence channels feeding TRUST
#   • Process  (transparency / explainability)
#   • Purpose  (perceived alignment with user priorities)
#
# All helpers here are stateless utilities.  Generator imports
# them to keep the main loop lean.
# -----------------------------------------------------------
from __future__ import annotations
import numpy as np
from .baseTrust_state import clip01  # tiny helper (min-max to [0,1])


# ===========================================================
# 1) PROCESS  – Explainability / transparency cues
# ===========================================================
def sample_xai_timing(
    rng: np.random.Generator,
    xai_present: np.ndarray,
    *,
    pre_rate: float,
    post_rate: float,
) -> np.ndarray:
    """
    Returns array of 0/1/2 per event:
        0 = no explanation
        1 = PRE       (shown before decision)
        2 = POST      (shown after decision)
    Probabilities are conditional on xai_present == 1.
    """
    timing = np.zeros_like(xai_present, dtype=int)
    mask = xai_present == 1
    if not mask.any():
        return timing

    # robust normalisation if (pre_rate + post_rate) ≠ 1.0
    s = max(pre_rate + post_rate, 1e-12)
    pre_p = pre_rate / s
    draw = rng.random(mask.sum())

    timing[mask] = np.where(draw < pre_p, 1, 2)
    return timing


def compute_proc_score(
    xai_present: np.ndarray,
    detail: np.ndarray,
    specificity: np.ndarray,
    consistency: np.ndarray,
) -> np.ndarray:
    """
    Lee & See 'Process' weighted quality score in [0,1].

    score = xai_present *
            (0.35·detail + 0.35·specificity + 0.30·consistency)
    """
    return xai_present.astype(float) * (
        0.35 * detail + 0.35 * specificity + 0.30 * consistency
    )


# ===========================================================
# 2) PURPOSE – Perceived alignment with priority KPI
# ===========================================================
def sample_user_profiles(
    rng: np.random.Generator,
    n_users: int,
    n_profiles: int,
) -> np.ndarray:
    """
    Assign each user to a simplified priority-profile bucket.
    """
    return rng.integers(0, n_profiles, size=n_users)


def sample_purpose_align(
    cfg,
    rng: np.random.Generator,
    *,
    profile_id: int,
    stake: float,
    time_pressure: float,
    xai_present: int,
    xai_faithfulness: float,
) -> float:
    """
    Scalar proxy in [0,1] for *perceived* alignment:
        0   = misaligned with user priority
        0.5 = partially aligned
        1   = clearly aligned

    Uses cfg fields:
        purpose_state_probs
        purpose_low_mean / mid_mean / high_mean
        purpose_noise_std
    """
    # --- choose perception state (0/1/2) --------------------
    state = rng.choice([0, 1, 2], p=cfg.purpose_state_probs)
    base = (
        cfg.purpose_low_mean
        if state == 0 else
        cfg.purpose_mid_mean
        if state == 1 else
        cfg.purpose_high_mean
    )

    # --- small contextual adjustments -----------------------
    adj = (
        0.06 * float(xai_present) * float(xai_faithfulness)   # faithful XAI ↑ alignment
        - 0.03 * stake                                        # high stake → user pickier
        - 0.04 * time_pressure                                # rushed → trust alignment less
    )

    val = base + adj + rng.normal(0.0, cfg.purpose_noise_std)
    return clip01(float(val))



# ===========================================================
# 3) PERFORMANCE – competence evidence (optional helper)
# ===========================================================
def performance_update(
    trust_T: float,
    *,
    ai_correct: bool,
    stake: float,
    alpha_pos: float,
    alpha_neg: float,
    eta_risk: float,
) -> float:
    """
    Return the performance-updated trust value (Lee & See ‘Competence’).
    """
    if ai_correct:
        return clip01(trust_T + alpha_pos * (1 - trust_T))
    else:
        return clip01(trust_T - alpha_neg * trust_T * (1 + eta_risk * stake))