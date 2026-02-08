from __future__ import annotations
import numpy as np


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fuse_beliefs_logodds(
    p_ai: np.ndarray,
    p_h: np.ndarray,
    trust_T: np.ndarray,
    agree: np.ndarray,
    lambda_pos: float,
    lambda_neg: float,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    Latent fusion: subjective belief p_subj uses trust + confirmation bias.
    agreement => overweight AI evidence; disagreement => underweight AI evidence.
    """
    w_ai = trust_T * np.where(agree == 1, lambda_pos, lambda_neg)
    w_h = (1.0 - trust_T)

    z = kappa * (w_ai * logit(p_ai) + w_h * logit(p_h))
    return sigmoid(z)
