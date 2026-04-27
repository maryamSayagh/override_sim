# override_sim/sim/generator.py
# -----------------------------------------------------------------
# SIMULATOR  – orchestrates one synthetic recommendation stream
# Uses helpers grouped by conceptual blocks.
# -----------------------------------------------------------------
from __future__ import annotations

from dataclasses import asdict
from typing import List

import numpy as np
import pandas as pd

from ..config.schemas import SimConfig

# ----- Signals & beliefs --------------------------------------------------
from .signals import sample_prob_binary, rec_from_prob
from .belief_block import fuse_beliefs_logodds

# ----- Utility (U-block) --------------------------------------------------
from .utility_blocks import (
    cost_fp,
    cost_fn,
    expected_cost,
    override_effort,
)

# ----- Incentives (I-block) ----------------------------------------------
from .incentive_block import (
    blame_penalty_accept,
    apply_observability,        # audit / policy masking + noise
)

# ----- Trust evidence channels (Process + Purpose + Performance helper) ---
from .trust_channels import (
    sample_xai_timing,
    compute_proc_score,
    sample_user_profiles,
    sample_purpose_align,
    performance_update,
)

# ----- Persistent latent trust -------------------------------------------
from .baseTrust_state import UserBaseState, clip01

# ----- Choice model sigmoid & intercept solver ----------------------------
from .override_intercept import sigmoid, tune_intercept_for_rate




class Simulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------
    def _init_users(self) -> List[UserBaseState]:
        a, b = self.cfg.trust_init_a, self.cfg.trust_init_b
        init_T = self.rng.beta(a, b, size=self.cfg.n_users)
        return [UserBaseState(float(t)) for t in init_T]

    # ---------------------------------------------------------------
    # Main run
    # ---------------------------------------------------------------
    def run(self, *, return_latents: bool = True) -> pd.DataFrame:
        cfg, rng = self.cfg, self.rng
        N = cfg.n_events

        # 1) Draw user ids & context
        user_id = rng.integers(0, cfg.n_users, size=N)
        priority_profile = sample_user_profiles(rng, cfg.n_users, cfg.n_priority_profiles)

        stake = rng.beta(2, 2, size=N)
        time_pressure = rng.beta(2, 2, size=N)

        policy_true = rng.beta(2, 2, size=N)                 # continuous
        audit_true  = rng.binomial(1, cfg.audit_rate, size=N)

        # 2) Ground truth & signals
        y_true = rng.binomial(1, cfg.base_rate_pi, size=N)

        ai_prob = sample_prob_binary(rng, y_true, cfg.mu_ai, stake, time_pressure, policy_true)
        ai_rec  = rec_from_prob(ai_prob, 0.5)

        h_prob  = sample_prob_binary(rng, y_true, cfg.mu_h, stake, time_pressure, policy_true)
        h_rec   = rec_from_prob(h_prob, 0.5)

        # 3) Explainability
        xai_present = rng.binomial(1, cfg.xai_present_rate, size=N)
        xai_timing  = sample_xai_timing(rng, xai_present,
                                        pre_rate=cfg.xai_pre_rate,
                                        post_rate=cfg.xai_post_rate)

        beta_q = lambda m: rng.beta(cfg.xai_q_a, cfg.xai_q_b, size=m.sum())
        xai_detail = np.zeros(N);  xai_specificity = np.zeros(N)
        xai_consistency = np.zeros(N);  xai_faithfulness = np.zeros(N)

        m = xai_present == 1
        if m.any():
            xai_detail[m]       = beta_q(m)
            xai_specificity[m]  = beta_q(m)
            xai_consistency[m]  = beta_q(m)
            xai_faithfulness[m] = beta_q(m)

        proc_score = compute_proc_score(
            xai_present, xai_detail, xai_specificity, xai_consistency
        )

        # 4) Purpose alignment proxy
        purpose_align = np.array([
            sample_purpose_align(
                cfg, rng,
                profile_id=int(priority_profile[u]),
                stake=float(stake[t]),
                time_pressure=float(time_pressure[t]),
                xai_present=int(xai_present[t]),
                xai_faithfulness=float(xai_faithfulness[t]),
            )
            for t, u in enumerate(user_id)
        ])

        # 5) Cost & effort arrays
        c_fp_arr = cost_fp(stake, cfg.c_fp, cfg.gamma_fp)
        c_fn_arr = cost_fn(stake, cfg.c_fn, cfg.gamma_fn)
        effort   = override_effort(time_pressure, cfg.effort_e0, cfg.effort_e1)

        # 6) Per-user states
        users = self._init_users()

        trust_T_latent = np.zeros(N)
        delta_latent   = np.zeros(N)

        util_lat, inc_lat, belief_lat = np.zeros(N), np.zeros(N), np.zeros(N)

        # 7) Event loop
        for t in range(N):
            u = int(user_id[t])
            usr = users[u]

            # (a) Performance backbone
            T_base = performance_update(
                usr.trust_T,
                ai_correct=(ai_rec[t] == y_true[t]),
                stake=stake[t],
                alpha_pos=cfg.alpha_pos,
                alpha_neg=cfg.alpha_neg,
                eta_risk=cfg.eta_risk,
            )

            # (b) Process & Purpose evidence
            proc_gain = cfg.w_proc * (
                0.5 * proc_score[t] if xai_timing[t] == 1             # pre
                else cfg.beta_post * proc_score[t] if xai_timing[t] == 2
                else 0.0
            )
            purp_shift = cfg.w_purp * (purpose_align[t] - 0.5)

            usr.trust_T = clip01(T_base + cfg.rho_trust * (proc_gain + purp_shift))
            trust_T_latent[t] = usr.trust_T

            # (c) Subjective belief
            agree = int(ai_rec[t] == h_rec[t])
            p_subj = fuse_beliefs_logodds(
                np.array([ai_prob[t]]), np.array([h_prob[t]]),
                np.array([usr.trust_T]), np.array([agree]),
                lambda_pos=cfg.lambda_pos, lambda_neg=cfg.lambda_neg
            )[0]

            # (d) Utility & incentives
            V_accept = -expected_cost(np.array([ai_rec[t]]), np.array([p_subj]),
                                      np.array([c_fp_arr[t]]), np.array([c_fn_arr[t]]))[0]

            V_ovr = -expected_cost(np.array([h_rec[t]]), np.array([p_subj]),
                                   np.array([c_fp_arr[t]]), np.array([c_fn_arr[t]]))[0] - effort[t]

            utility_term   = V_ovr - V_accept
            incentive_term = blame_penalty_accept(
                 p_ai=np.array([ai_prob[t]]),
                ai_rec=np.array([ai_rec[t]]),
                audit=np.array([audit_true[t]]),
                policy_strict=np.array([policy_true[t]]),
                stake=np.array([stake[t]]),
                blame_scale=cfg.blame_scale,)[0]

            belief_term = (
                -cfg.w_proc_now * proc_score[t] *
                (1 - cfg.gamma_stress * time_pressure[t])
                if xai_timing[t] == 1 else 0.0
            )

            delta = (
                cfg.w_B * belief_term +
                cfg.w_U * utility_term +
                cfg.w_I * incentive_term
            )

            delta_latent[t] = delta
            util_lat[t], inc_lat[t], belief_lat[t] = utility_term, incentive_term, belief_term

        # 8) Override sampling
        b_int = tune_intercept_for_rate(delta_latent, cfg.target_override_rate, cfg.kappa)
        p_ovr = sigmoid(cfg.kappa * delta_latent + b_int)
        override = rng.binomial(1, p_ovr, size=N)

        # 9) Apply observability mask
        audit_obs, policy_obs = apply_observability(
            cfg, rng, audit_true.astype(float), policy_true.astype(float)
        )

        # 10) Assemble DataFrame
        df = pd.DataFrame(
            {
                "t": np.arange(N, dtype=int),
                "user_id": user_id.astype(int),
                "stake": stake,
                "time_pressure": time_pressure,
                "policy_strict": policy_obs,
                "audit": audit_obs,
                "y_true": y_true,
                "ai_prob": ai_prob,
                "ai_rec": ai_rec,
                "xai_present": xai_present,
                "xai_timing": xai_timing,
                "xai_detail": xai_detail,
                "xai_specificity": xai_specificity,
                "xai_consistency": xai_consistency,
                "xai_faithfulness": xai_faithfulness,
                "purpose_align": purpose_align,
                "override": override,
                "p_override_true": p_ovr,
            }
        )

        if return_latents:
            df["human_prob_latent"]  = h_prob
            df["human_rec_latent"]   = h_rec
            df["trust_T_latent"]     = trust_T_latent
            df["delta_latent"]       = delta_latent
            df["proc_score_latent"]  = proc_score
            df["utility_term_latent"]   = util_lat
            df["incentive_term_latent"] = inc_lat
            df["belief_term_latent"]    = belief_lat
            df["intercept_b"]        = float(b_int)

        for k, v in asdict(cfg).items():
            df.attrs[f"cfg.{k}"] = v
        df.attrs["cfg.intercept_b"] = float(b_int)
        df.attrs["override_rate"]   = float(df["override"].mean())

        return df
    