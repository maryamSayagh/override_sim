# override_sim/config/worlds.py
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

from override_sim.config.presets import phase1
from override_sim.config.schemas import SimConfig  # if your SimConfig lives elsewhere, adjust import


def build_worlds(target_override_rate: float = 0.25) -> Dict[str, Tuple[SimConfig, str]]:
    """
    Returns {world_name: (cfg, note)}. All worlds share the same override prevalence target.
    """
    base = phase1()
    base = replace(base, target_override_rate=target_override_rate)

    worlds: Dict[str, Tuple[SimConfig, str]] = {}

    # W0 — baseline
    worlds["W0_baseline"] = (base, "Phase1 baseline; control world")

    # W1 — utility dominant: higher cost pressure + lower effort
    worlds["W1_utility_dominant"] = (
        replace(
            base,
            c_fp=1.2,
            c_fn=3.0,
            gamma_fp=1.1,
            gamma_fn=2.0,
            effort_e0=0.08,
            effort_e1=0.40,
        ),
        "Higher cost pressure; lower override effort -> U dominates",
    )

    # W2 — trust/bias dominant: stronger algorithm aversion + confirmation bias
    worlds["W2_trust_bias_dominant"] = (
        replace(
            base,
            alpha_pos=0.01,
            alpha_neg=0.25,
            lambda_pos=1.6,
            lambda_neg=0.35,
        ),
        "Stronger aversion & confirmation bias -> B dominates",
    )

    # W3 — incentives dominant: higher audit + stronger blame penalties
    worlds["W3_incentives_dominant"] = (
        replace(
            base,
            audit_rate=0.55,
            blame_scale=6.0,
        ),
        "Audit/accountability regime -> I dominates",
    )

    return worlds
