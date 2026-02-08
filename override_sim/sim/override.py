from __future__ import annotations
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def tune_intercept_for_rate(
    deltas: np.ndarray,
    target_rate: float,
    kappa: float,
    lo: float = -20.0,
    hi: float = 20.0,
    iters: int = 60,
) -> float:
    """
    Find b such that mean(sigmoid(kappa*deltas + b)) ~= target_rate.
    Bisection on b.
    """
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        rate = float(np.mean(sigmoid(kappa * deltas + mid)))
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
