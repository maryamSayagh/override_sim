from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    # Data volume
    n_users: int = 100
    n_events: int = 50_000
    base_rate_pi: float = 0.30  # P(y_true=1)
    seed: int = 42

    # AI / Human signal strength (higher => more accurate)
    mu_ai: float = 2.0
    mu_h: float = 1.3

    # Trust update (algorithm aversion: alpha_neg > alpha_pos)
    alpha_pos: float = 0.015
    alpha_neg: float = 0.15
    trust_init_a: float = 6.0  # Beta prior
    trust_init_b: float = 4.0  # => mean ~0.6

    # Confirmation bias parameters (agreement overweight, disagreement underweight)
    lambda_pos: float = 1.3
    lambda_neg: float = 0.5

    # Costs (utility/risk)
    c_fp: float = 1.0
    c_fn: float = 2.0
    gamma_fp: float = 1.0
    gamma_fn: float = 1.5

    # Override effort (higher => fewer overrides)
    effort_e0: float = 0.10
    effort_e1: float = 0.50

    # Incentives
    audit_rate: float = 0.30
    blame_scale: float = 4.0

    # Override sampling (choice sharpness + target override rate)
    kappa: float = 6.0
    target_override_rate: float = 0.35
