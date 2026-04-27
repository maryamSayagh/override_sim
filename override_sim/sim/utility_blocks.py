from __future__ import annotations
import numpy as np


def cost_fp(stake: np.ndarray, c_fp: float, gamma_fp: float) -> np.ndarray:
    return c_fp * (1.0 + gamma_fp * stake)


def cost_fn(stake: np.ndarray, c_fn: float, gamma_fn: float) -> np.ndarray:
    return c_fn * (1.0 + gamma_fn * stake)


def expected_cost(
    decision: np.ndarray,  # 0 or 1
    p_subj: np.ndarray,    # P(y=1)
    c_fp_arr: np.ndarray,
    c_fn_arr: np.ndarray,
) -> np.ndarray:
    """
    If decision=1, error occurs when y=0 with prob (1-p_subj) and costs FP.
    If decision=0, error occurs when y=1 with prob p_subj and costs FN.
    """
    return np.where(
        decision == 1,
        (1.0 - p_subj) * c_fp_arr,
        p_subj * c_fn_arr,
    )


def override_effort(time_pressure: np.ndarray, e0: float, e1: float) -> np.ndarray:
    return e0 + e1 * time_pressure
