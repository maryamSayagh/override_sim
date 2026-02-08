from __future__ import annotations
from dataclasses import asdict
import numpy as np
import pandas as pd

from ..config.schemas import SimConfig
from .user_state import UserState
from .signals import sample_prob_binary, rec_from_prob
from .belief import fuse_beliefs_logodds
from .costs import cost_fp, cost_fn, expected_cost, override_effort
from .incentives import blame_penalty_accept
from .override import sigmoid, tune_intercept_for_rate


class Simulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _init_users(self) -> list[UserState]:
        # Beta prior mean for trust init
        a, b = self.cfg.trust_init_a, self.cfg.trust_init_b
        init_trust = self.rng.beta(a, b, size=self.cfg.n_users)
        return [UserState(float(t)) for t in init_trust]

    def run(self, return_latents: bool = True) -> pd.DataFrame:
        cfg = self.cfg
        rng = self.rng

        # Assign user per event
        user_id = rng.integers(0, cfg.n_users, size=cfg.n_events)

        # Context
        stake = rng.beta(2, 2, size=cfg.n_events)
        time_pressure = rng.beta(2, 2, size=cfg.n_events)
        policy_strict = rng.beta(2, 2, size=cfg.n_events)
        audit = rng.binomial(1, cfg.audit_rate, size=cfg.n_events)

        # Ground truth
        y_true = rng.binomial(1, cfg.base_rate_pi, size=cfg.n_events)

        # AI & Human signals
        ai_prob = sample_prob_binary(rng, y_true, cfg.mu_ai, stake, time_pressure, policy_strict)
        ai_rec = rec_from_prob(ai_prob, 0.5)

        human_prob = sample_prob_binary(rng, y_true, cfg.mu_h, stake, time_pressure, policy_strict)
        human_rec = rec_from_prob(human_prob, 0.5)

        # Per-user trust state evolves over time (sequential simulation)
        users = self._init_users()
        trust_T = np.zeros(cfg.n_events, dtype=float)

        # We'll compute override deltas first, then tune intercept to target override rate
        deltas = np.zeros(cfg.n_events, dtype=float)

        # Precompute costs
        c_fp_arr = cost_fp(stake, cfg.c_fp, cfg.gamma_fp)
        c_fn_arr = cost_fn(stake, cfg.c_fn, cfg.gamma_fn)
        effort = override_effort(time_pressure, cfg.effort_e0, cfg.effort_e1)

        # Sequential loop (needed for trust updates)
        for t in range(cfg.n_events):
            u = int(user_id[t])
            T = users[u].trust_T
            trust_T[t] = T

            agree = 1 if ai_rec[t] == human_rec[t] else 0

            p_subj = fuse_beliefs_logodds(
                p_ai=np.array([ai_prob[t]]),
                p_h=np.array([human_prob[t]]),
                trust_T=np.array([T]),
                agree=np.array([agree]),
                lambda_pos=cfg.lambda_pos,
                lambda_neg=cfg.lambda_neg,
                kappa=1.0,
            )[0]

            # Utility: expected costs if accept AI vs override to human_rec
            V_accept = -expected_cost(
                decision=np.array([ai_rec[t]]),
                p_subj=np.array([p_subj]),
                c_fp_arr=np.array([c_fp_arr[t]]),
                c_fn_arr=np.array([c_fn_arr[t]]),
            )[0]

            V_override = -expected_cost(
                decision=np.array([human_rec[t]]),
                p_subj=np.array([p_subj]),
                c_fp_arr=np.array([c_fp_arr[t]]),
                c_fn_arr=np.array([c_fn_arr[t]]),
            )[0] - effort[t]

            # Incentives: blame penalty for accepting AI
            blame = blame_penalty_accept(
                p_ai=np.array([ai_prob[t]]),
                ai_rec=np.array([ai_rec[t]]),
                audit=np.array([audit[t]]),
                policy_strict=np.array([policy_strict[t]]),
                stake=np.array([stake[t]]),
                blame_scale=cfg.blame_scale,
            )[0]
            V_accept_tot = V_accept - blame
            V_override_tot = V_override

            deltas[t] = V_override_tot - V_accept_tot

            # Update trust with true correctness (Phase 1 simplification)
            ai_correct = (ai_rec[t] == y_true[t])
            users[u].update(ai_correct=bool(ai_correct), alpha_pos=cfg.alpha_pos, alpha_neg=cfg.alpha_neg)

        # Tune intercept b to hit ~target override rate
        intercept_b = tune_intercept_for_rate(deltas, cfg.target_override_rate, cfg.kappa)

        # Sample overrides
        p_override = sigmoid(cfg.kappa * deltas + intercept_b)
        override = rng.binomial(1, p_override, size=cfg.n_events)

        # Confusion type
        confusion = np.where(
            (ai_rec == 1) & (y_true == 1), "TP",
            np.where(
                (ai_rec == 0) & (y_true == 0), "TN",
                np.where(
                    (ai_rec == 1) & (y_true == 0), "FP",
                    "FN"
                )
            )
        )

        df = pd.DataFrame({
            "t": np.arange(cfg.n_events, dtype=int),
            "user_id": user_id.astype(int),
            "stake": stake.astype(float),
            "time_pressure": time_pressure.astype(float),
            "policy_strict": policy_strict.astype(float),
            "audit": audit.astype(int),

            "y_true": y_true.astype(int),
            "ai_prob": ai_prob.astype(float),
            "ai_rec": ai_rec.astype(int),

            "override": override.astype(int),
            "p_override_true": p_override.astype(float),  # for analysis only
            "confusion_type": confusion,
        })

        if return_latents:
            df["human_prob_latent"] = human_prob.astype(float)
            df["human_rec_latent"] = human_rec.astype(int)
            df["trust_T_latent"] = trust_T.astype(float)
            df["delta_latent"] = deltas.astype(float)
            df["intercept_b"] = float(intercept_b)

        # Attach config snapshot for reproducibility (optional)
        for k, v in asdict(cfg).items():
            df.attrs[f"cfg.{k}"] = v
        df.attrs["cfg.intercept_b"] = float(intercept_b)
        df.attrs["override_rate"] = float(df["override"].mean())

        return df
