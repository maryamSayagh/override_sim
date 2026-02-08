from __future__ import annotations
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def sample_prob_binary(
    rng: np.random.Generator,
    y_true: np.ndarray,
    mu: float,
    stake: np.ndarray,
    time_pressure: np.ndarray,
    policy_strict: np.ndarray,
) -> np.ndarray:
    """
    Generates probabilities p in (0,1) that are correlated with y_true.
    mu controls signal strength (accuracy).
    Context influences difficulty a little (optional).
    """
    # map y_true {0,1} -> {-1,+1}
    s = 2.0 * y_true - 1.0

    # context effects (small)
    ctx = 0.25 * (stake - 0.5) - 0.15 * (time_pressure - 0.5) - 0.10 * (policy_strict - 0.5)

    z = mu * s + ctx + rng.normal(0.0, 1.0, size=y_true.shape[0])
    return sigmoid(z)


def rec_from_prob(p: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (p >= threshold).astype(int)
