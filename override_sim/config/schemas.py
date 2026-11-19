from __future__ import annotations
from dataclasses import dataclass

"""
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
    target_override_rate: float = 0.25

    # --- Lee & See (2004) trust evidence weights ---
    # T_{t+1} = (1-rho)T_t + rho*(w_perf*f_perf + w_proc*f_proc + w_purp*f_purp)
    rho_trust: float = 0.06
    w_perf: float = 0.55
    w_proc: float = 0.25
    w_purp: float = 0.20

    # Performance calibration / algorithm aversion
    # f_perf = c - lambda_err*(1-c)*(1 + eta_risk*stake)
    lambda_err: float = 1.0
    eta_risk: float = 0.8

    # Process timing: post explanations update trust less than pre explanations
    beta_post: float = 0.4

    # Stress/workload reduces how much users benefit from explanation at decision time
    gamma_stress: float = 0.6

    # How much PRE explanation reduces override directly at decision time
    w_proc_now: float = 0.35

    # Utility/Risk costs
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
    target_override_rate: float = 0.25

    # --- Explainability (Process) generator knobs ---
    # Probability any explanation is available
    xai_present_rate: float = 0.75

    # Timing probabilities (conditional on xai_present=1)
    # must sum to 1.0
    xai_pre_rate: float = 0.55
    xai_post_rate: float = 0.45

    # Distribution shapes for explanation quality in [0,1]
    # beta(a,b) for detail/specificity/consistency/faithfulness
    xai_q_a: float = 3.0
    xai_q_b: float = 2.0

    # Purpose/alignment (latent but we can also log proxy if you want)
    purpose_a: float = 2.5
    purpose_b: float = 2.5



"""

@dataclass
class SimConfig:
    # Data volume
    n_users: int = 100
    n_events: int = 50_000
    base_rate_pi: float = 0.30  # P(y_true=1)
    seed: int = 42

    # AI / Human signal strength (higher => more accurate)
    mu_ai: float = 1
    mu_h: float = 0.7

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
    target_override_rate: float = 0.25

    # Initial trust prior
    trust_init_a: float = 6.0  # Beta prior
    trust_init_b: float = 4.0  # => mean ~0.6

    # --- Lee & See (2004): Trust = f(Performance, Process, Purpose) ---

    # Performance (competence / observed reliability)
    # Implemented through the base trust update:
    # correct: T + alpha_pos * (1 - T)
    # wrong:   T - alpha_neg * T * (1 + eta_risk * stake)
    #
    # alpha_neg > alpha_pos encodes algorithm aversion / negativity asymmetry
    alpha_pos: float = 0.015
    alpha_neg: float = 0.15
    eta_risk: float = 0.8

    # Optional performance helper parameters
    # Keep only if still used in generator code paths
    w_perf: float = 0.55
    lambda_err: float = 1.0

    # Process (transparency / predictability via explainability)
    # Process influences trust through proc_gain and also affects the current decision
    rho_trust: float = 0.06
    w_proc: float = 0.25
    beta_post: float = 0.4

    # Stress/workload reduces how much users benefit from explanation at decision time
    gamma_stress: float = 0.6

    # How much PRE explanation reduces override directly at decision time
    w_proc_now: float = 0.35

    # Purpose (alignment with user goals)
    # Purpose is modeled as perceived validation of the decision-maker's priority,
    # not objective global optimality.
    # It affects trust through:
    # purp_shift = w_purp * (purpose_align - 0.5)
    w_purp: float = 0.20

    # --- Explainability (Process) generator knobs ---
    # Probability any explanation is available
    xai_present_rate: float = 0.75

    # Timing probabilities (conditional on xai_present=1)
    # must sum to 1.0
    xai_pre_rate: float = 0.55
    xai_post_rate: float = 0.45

    # Distribution shapes for explanation quality in [0,1]
    # beta(a,b) for detail/specificity/consistency/faithfulness
    xai_q_a: float = 3.0
    xai_q_b: float = 2.0

    # --- Purpose generator knobs ---
    # We keep this intentionally simple:
    # - each user has one simplified priority profile
    # - each event has a perceived level of satisfaction for that profile
    # - purpose_align is a scalar proxy in [0,1]
    #
    # It is NOT meant to simulate full multi-KPI optimization.
    # It is a scoped proxy for perceived alignment with what the decision-maker cares about.

    # Number of simplified priority profiles
    n_priority_profiles: int = 3

    # Probability a recommendation is perceived as:
    # 0 = misaligned
    # 1 = partially aligned
    # 2 = clearly aligned
    purpose_state_probs: tuple[float, float, float] = (0.30, 0.40, 0.30)

    # Base purpose means for the 3 simplified states
    purpose_low_mean: float = 0.20
    purpose_mid_mean: float = 0.55
    purpose_high_mean: float = 0.85

    # Noise around each state mean
    purpose_noise_std: float = 0.08

    # Optional Beta fallback if you still use older purpose sampling paths
    purpose_a: float = 2.5
    purpose_b: float = 2.5

    # --- Multi-world control weights ---
    # Global weights used to control which mechanism dominates override
    w_B: float = 1.0
    w_U: float = 1.0
    w_I: float = 1.0

    # --- Observability / noise knobs ---
    # Used to make some regimes less "too clean to be true"
    incentive_noise_std: float = 0.0
    history_noise_std: float = 0.0
    hide_audit_prob: float = 0.0
    hide_policy_prob: float = 0.0