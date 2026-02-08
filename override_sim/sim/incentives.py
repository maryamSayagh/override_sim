from __future__ import annotations
import numpy as np


def ai_wrong_proxy(p_ai: np.ndarray, ai_rec: np.ndarray) -> np.ndarray:
    # If AI recommends 1, "wrongness" proxy is 1 - p_ai; if recommends 0, proxy is p_ai
    return np.where(ai_rec == 1, 1.0 - p_ai, p_ai)


def blame_penalty_accept(
    p_ai: np.ndarray,
    ai_rec: np.ndarray,
    audit: np.ndarray,
    policy_strict: np.ndarray,
    stake: np.ndarray,
    blame_scale: float,
) -> np.ndarray:
    # Penalty increases with audit*policy*stake and perceived chance AI is wrong
    return blame_scale * audit * policy_strict * stake * ai_wrong_proxy(p_ai, ai_rec)
