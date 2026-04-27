# override_sim/config/worlds.py
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List

from override_sim.config.schemas import SimConfig


@dataclass
class WorldSpec:
    name: str
    description: str
    overrides: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Baseline world assumptions:
# - same target override rate (calibrated inside generator)
# - same sample size
# - only behavioral regime changes
# ---------------------------------------------------------------------

WORLD_SPECS: List[WorldSpec] = [
    WorldSpec(
        name="W0_baseline",
        description="Balanced world: trust, utility, incentives all active at moderate levels.",
        overrides={
            # Trust / belief / process
            "lambda_pos": 1.10,
            "lambda_neg": 1.45,
            "rho_trust": 0.18,
            "w_proc": 0.28,
            "w_proc_now": 0.22,
            "w_purp": 0.16,
            "beta_post": 0.20,
            "gamma_stress": 0.35,
            "eta_risk": 0.30,

            # Incentives / effort / choice
            "blame_scale": 0.55,
            "effort_e0": 0.10,
            "effort_e1": 0.40,
            "kappa": 1.00,

            # Explainability
            "xai_present_rate": 0.55,
            "xai_pre_rate": 0.60,
            "xai_post_rate": 0.40,  # should complement xai_pre_rate

            # World weights
            "w_B": 1.00,
            "w_U": 1.00,
            "w_I": 1.00,

            # Observability
            "incentive_noise_std": 0.00,
            "history_noise_std": 0.00,
            "hide_audit_prob": 0.00,
            "hide_policy_prob": 0.00,
        },
    ),
    WorldSpec(
        name="W1_utility_dominant",
        description="Override behavior is driven mostly by stake, effort, and utility tradeoffs.",
        overrides={
            # Trust / belief / process
            "lambda_pos": 1.00,
            "lambda_neg": 1.20,
            "rho_trust": 0.10,
            "w_proc": 0.12,
            "w_proc_now": 0.10,
            "w_purp": 0.08,
            "beta_post": 0.10,
            "gamma_stress": 0.45,
            "eta_risk": 0.90,

            # Incentives / effort / choice
            "blame_scale": 0.30,
            "effort_e0": 0.08,
            "effort_e1": 0.80,
            "kappa": 1.15,

            # Explainability
            "xai_present_rate": 0.40,
            "xai_pre_rate": 0.50,
            "xai_post_rate": 0.50,

            # World weights
            "w_B": 0.70,
            "w_U": 1.45,
            "w_I": 0.70,

            # Observability
            "incentive_noise_std": 0.05,
            "history_noise_std": 0.05,
            "hide_audit_prob": 0.00,
            "hide_policy_prob": 0.00,
        },
    ),
    WorldSpec(
        name="W2_trust_bias_dominant",
        description="Override behavior is driven mostly by trust dynamics, confirmation effects, and process cues.",
        overrides={
            # Trust / belief / process
            "lambda_pos": 1.25,
            "lambda_neg": 2.00,
            "rho_trust": 0.32,
            "w_proc": 0.50,
            "w_proc_now": 0.42,
            "w_purp": 0.26,
            "beta_post": 0.35,
            "gamma_stress": 0.22,
            "eta_risk": 0.20,

            # Incentives / effort / choice
            "blame_scale": 0.25,
            "effort_e0": 0.10,
            "effort_e1": 0.25,
            "kappa": 1.05,

            # Explainability
            "xai_present_rate": 0.80,
            "xai_pre_rate": 0.72,
            "xai_post_rate": 0.28,

            # World weights
            "w_B": 1.55,
            "w_U": 0.70,
            "w_I": 0.60,

            # Observability
            "incentive_noise_std": 0.12,
            "history_noise_std": 0.00,
            "hide_audit_prob": 0.10,
            "hide_policy_prob": 0.10,
        },
    ),
    WorldSpec(
        name="W3_incentives_dominant",
        description="Override behavior is driven mostly by audit, policy pressure, and blame avoidance.",
        overrides={
            # Trust / belief / process
            "lambda_pos": 0.95,
            "lambda_neg": 1.10,
            "rho_trust": 0.08,
            "w_proc": 0.08,
            "w_proc_now": 0.06,
            "w_purp": 0.06,
            "beta_post": 0.08,
            "gamma_stress": 0.40,
            "eta_risk": 0.15,

            # Incentives / effort / choice
            "audit_rate": 0.55,
            "blame_scale": 1.20,
            "effort_e0": 0.10,
            "effort_e1": 0.25,
            "kappa": 1.20,

            # Explainability
            "xai_present_rate": 0.35,
            "xai_pre_rate": 0.45,
            "xai_post_rate": 0.55,

            # World weights
            "w_B": 0.55,
            "w_U": 0.75,
            "w_I": 1.65,

            # Observability
            "incentive_noise_std": 0.15,
            "history_noise_std": 0.08,
            "hide_audit_prob": 0.20,
            "hide_policy_prob": 0.20,
        },
    ),
]


def get_world_names() -> List[str]:
    return [w.name for w in WORLD_SPECS]


def get_world_spec(name: str) -> WorldSpec:
    for w in WORLD_SPECS:
        if w.name == name:
            return w
    raise KeyError(f"Unknown world name: {name}")


def build_world_cfg(base_cfg: SimConfig, world_name: str) -> SimConfig:
    """
    Return a SimConfig dataclass instance for the requested world.
    """
    spec = get_world_spec(world_name)
    return replace(base_cfg, **spec.overrides)


def build_all_world_cfgs(base_cfg: SimConfig) -> Dict[str, SimConfig]:
    """
    Convenience helper: build all world configs from one base config.
    """
    return {
        spec.name: build_world_cfg(base_cfg, spec.name)
        for spec in WORLD_SPECS
    }


def get_world_description(world_name: str) -> str:
    return get_world_spec(world_name).description