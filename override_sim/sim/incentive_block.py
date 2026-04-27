# override_sim/sim/incentive_block.py
# ------------------------------------------------------------
# Incentive (I-block) helpers
#
# 1.  blame_penalty_accept(...)
#     → expected blame cost when the decision-maker accepts
#       the AI recommendation under audit / policy scrutiny.
#
# 2.  apply_observability(...)
#     → adds noise / missingness to audit & policy signals
#       so worlds can control observability realism.
# ------------------------------------------------------------
from __future__ import annotations

import numpy as np


# =========================================================================
# 1)  Blame penalty  (audit / policy pressure)
# =========================================================================
def blame_penalty_accept(
    *,
    p_ai: np.ndarray,        # AI confidence (probability AI rec == positive)
    ai_rec: np.ndarray,      # AI recommendation 0/1
    audit: np.ndarray,       # observed audit flag (true value)
    policy_strict: np.ndarray,  # observed policy strictness [0,1]
    stake: np.ndarray,       # stake [0,1]
    blame_scale: float,      # global multiplier
) -> np.ndarray:
    """
    Expected blame *cost* if the user accepts the AI recommendation.

    • Stronger when:
        - audit == 1
        - policy is strict
        - stake is high
        - AI confidence is low
    • Zero if audit == 0  AND  policy == 0.
    """
    # 1-p_ai is the AI’s own uncertainty; we penalise acting on low-confidence advice
    uncertainty = 1.0 - p_ai

    pressure = (audit.astype(float) + policy_strict) / 2.0   # ∈ [0,1]
    cost     = blame_scale * pressure * stake * uncertainty

    # If the user would override anyway (ai_rec ignored) this still represents
    # the hypothetical blame *risk* – caller decides how to combine.
    return cost


# =========================================================================
# 2)  Observability masking / noise
# =========================================================================
def apply_observability(
    cfg,
    rng: np.random.Generator,
    audit_true: np.ndarray,        # float array (0/1)
    policy_true: np.ndarray,       # float array [0,1]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Introduce realism by corrupting audit/policy before logging:

        • additive Gaussian noise  (cfg.incentive_noise_std)
        • random hiding            (cfg.hide_audit_prob, cfg.hide_policy_prob)

    Returns (audit_obs, policy_obs)

    NOTE: audit_true & policy_true should already be floats (not ints) when passed.
    """
    audit_obs  = audit_true.copy()
    policy_obs = policy_true.copy()

    # ----- Additive noise -----------------------------------
    if cfg.incentive_noise_std > 0.0:
        # Binary audit: add noise then threshold back to 0/1
        audit_obs = audit_obs + rng.normal(0.0, cfg.incentive_noise_std, size=audit_obs.shape)
        audit_obs = (audit_obs > 0.5).astype(float)

        # Continuous policy: add noise then clip
        policy_obs = np.clip(
            policy_obs + rng.normal(0.0, cfg.incentive_noise_std, size=policy_obs.shape),
            0.0,
            1.0,
        )

    # ----- Random hiding / missingness ----------------------
    if cfg.hide_audit_prob > 0.0:
        mask = rng.random(audit_obs.size) < cfg.hide_audit_prob
        audit_obs[mask] = np.nan

    if cfg.hide_policy_prob > 0.0:
        mask = rng.random(policy_obs.size) < cfg.hide_policy_prob
        policy_obs[mask] = np.nan

    return audit_obs, policy_obs