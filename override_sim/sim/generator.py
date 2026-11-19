
"""
 Synthetic Decision Datasets Simulator

Generates sequential human–AI decisions using:
    B: Belief and trust
    U: Utility and risk
    I: Institutional incentives

Outputs:
    Full dataset:
        Observable variables, override decisions,
        ground truth, final actions, internal probabilities,
        and optional latent states.

    HOVER dataset:
        Observable features and binary override target only.

Important:
    - Recommendations represent executable binary actions.
    - Overrides must change the recommended action.
    - Current ground truth cannot influence the current decision.
    - Outcome feedback updates trust for subsequent decisions.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List

import numpy as np
import pandas as pd

# -------------------------------------------------------
# Project imports
# -------------------------------------------------------

from override_sim.data.data_schema import (
    RAW_OBSERVABLE_COLS,
    TARGET_COL,
)

from override_sim.config.schemas import SimConfig

from .signals import sample_prob_binary, rec_from_prob
from .belief_block import fuse_beliefs_logodds

from .utility_blocks import (
    cost_fp,
    cost_fn,
    expected_cost,
    override_effort,
)

from .incentive_block import (
    blame_penalty_accept,
    apply_observability,
)

from .trust_channels import (
    sample_xai_timing,
    compute_proc_score,
    sample_user_profiles,
    sample_purpose_align,
    performance_update,
)

from .baseTrust_state import UserBaseState, clip01

from .override_intercept import (
    sigmoid,
    tune_intercept_for_rate,
)


class Simulator:

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    # ---------------------------------------------------
    # Initialize persistent user states
    # ---------------------------------------------------

    def _init_users(self) -> List[UserBaseState]:

        a = self.cfg.trust_init_a
        b = self.cfg.trust_init_b

        initial_trust = self.rng.beta(
            a,
            b,
            size=self.cfg.n_users,
        )

        return [
            UserBaseState(float(trust))
            for trust in initial_trust
        ]

    # ---------------------------------------------------
    # Calibrate effective override probability
    # ---------------------------------------------------

    def _override_probabilities(
        self,
        delta: np.ndarray,
        ai_rec: np.ndarray,
        human_rec: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Calibrate the overall override rate while requiring
        that an override changes the AI-recommended action.

        Only disagreements are eligible for overriding.
        """

        cfg = self.cfg

        eligible = ai_rec != human_rec
        eligible_rate = float(eligible.mean())

        target = float(cfg.target_override_rate)

        if not 0 < target < 1:
            raise ValueError(
                "target_override_rate must be between 0 and 1. "
                "Fix: Check the world configuration."
            )

        if eligible_rate <= target:
            raise ValueError(
                f"Target override rate ({target:.3f}) is not "
                f"feasible with disagreement rate "
                f"({eligible_rate:.3f}). "
                "Fix: Check AI/human signal generation or "
                "revise the target override rate based on "
                "the intended experimental conditions."
            )

        # Convert the global target into the required
        # override probability among eligible decisions.
        conditional_target = target / eligible_rate

        intercept = tune_intercept_for_rate(
            delta[eligible],
            conditional_target,
            cfg.kappa,
        )

        probabilities = np.zeros(
            len(delta),
            dtype=float,
        )

        probabilities[eligible] = sigmoid(
            cfg.kappa * delta[eligible] + intercept
        )

        return probabilities, float(intercept)

    # ---------------------------------------------------
    # Main simulation
    # ---------------------------------------------------

    def run(
        self,
        *,
        return_latents: bool = True,
        hover_mode: bool = False,
    ) -> pd.DataFrame:

        cfg = self.cfg
        rng = self.rng
        N = cfg.n_events

        if N <= 0:
            raise ValueError(
                "n_events must be positive. "
                "Fix: Check the simulation configuration."
            )

        # ------------------------------------------------
        # 1. Users and decision context
        # ------------------------------------------------

        user_id = rng.integers(
            0,
            cfg.n_users,
            size=N,
        )

        priority_profile = sample_user_profiles(
            rng,
            cfg.n_users,
            cfg.n_priority_profiles,
        )

        stake = rng.beta(2, 2, size=N)
        time_pressure = rng.beta(2, 2, size=N)

        policy_true = rng.beta(2, 2, size=N)

        audit_true = rng.binomial(
            1,
            cfg.audit_rate,
            size=N,
        )

        # ------------------------------------------------
        # 2. Ground truth and recommendations
        # ------------------------------------------------

        y_true = rng.binomial(
            1,
            cfg.base_rate_pi,
            size=N,
        )

        ai_prob = sample_prob_binary(
            rng,
            y_true,
            cfg.mu_ai,
            stake,
            time_pressure,
            policy_true,
        )

        ai_rec = rec_from_prob(ai_prob, 0.5)

        h_prob = sample_prob_binary(
            rng,
            y_true,
            cfg.mu_h,
            stake,
            time_pressure,
            policy_true,
        )

        h_rec = rec_from_prob(h_prob, 0.5)


      
  
        # ------------------------------------------------
        # 3. Explainability
        # ------------------------------------------------

        xai_present = rng.binomial(
            1,
            cfg.xai_present_rate,
            size=N,
        )

        xai_timing = sample_xai_timing(
            rng,
            xai_present,
            pre_rate=cfg.xai_pre_rate,
            post_rate=cfg.xai_post_rate,
        )

        xai_detail = np.zeros(N)
        xai_specificity = np.zeros(N)
        xai_consistency = np.zeros(N)
        xai_faithfulness = np.zeros(N)

        present = xai_present == 1

        if present.any():

            def sample_quality():
                return rng.beta(
                    cfg.xai_q_a,
                    cfg.xai_q_b,
                    size=present.sum(),
                )

            xai_detail[present] = sample_quality()
            xai_specificity[present] = sample_quality()
            xai_consistency[present] = sample_quality()
            xai_faithfulness[present] = sample_quality()

            proc_score = compute_proc_score(
            xai_present,
            xai_detail,
            xai_specificity,
            xai_consistency,
        )

        # ------------------------------------------------
        # 4. Purpose alignment
        # ------------------------------------------------

        purpose_align = np.array([
            sample_purpose_align(
                cfg,
                rng,
                profile_id=int(priority_profile[u]),
                stake=float(stake[t]),
                time_pressure=float(time_pressure[t]),
                xai_present=int(xai_present[t]),
                xai_faithfulness=float(
                    xai_faithfulness[t]
                ),
            )
            for t, u in enumerate(user_id)
        ])

        # ------------------------------------------------
        # 5. Utility and effort
        # ------------------------------------------------

        c_fp_arr = cost_fp(
            stake,
            cfg.c_fp,
            cfg.gamma_fp,
        )

        c_fn_arr = cost_fn(
            stake,
            cfg.c_fn,
            cfg.gamma_fn,
        )

        effort = override_effort(
            time_pressure,
            cfg.effort_e0,
            cfg.effort_e1,
        )

        # ------------------------------------------------
        # 6. Persistent user states and latent arrays
        # ------------------------------------------------

        users = self._init_users()

        trust_T_latent = np.zeros(N)
        delta_latent = np.zeros(N)

        utility_latent = np.zeros(N)
        incentive_latent = np.zeros(N)
        belief_latent = np.zeros(N)

        # ------------------------------------------------
        # 7. Sequential decision loop
        # ------------------------------------------------

        for t in range(N):

            user = users[int(user_id[t])]

            # --------------------------------------------
            # A. Trust available at decision time
            # --------------------------------------------

            # Previous performance feedback is already
            # reflected in user.trust_T.

            # Process and purpose provide current evidence.

            proc_gain = cfg.w_proc * (
                0.5 * proc_score[t]
                if xai_timing[t] == 1
                else (
                    cfg.beta_post * proc_score[t]
                    if xai_timing[t] == 2
                    else 0.0
                )
            )

            purp_shift = (
                cfg.w_purp
                * (purpose_align[t] - 0.5)
            )

            user.trust_T = clip01(
                user.trust_T
                + cfg.rho_trust * (
                    proc_gain + purp_shift
                )
            )

            trust_T_latent[t] = user.trust_T

            # --------------------------------------------
            # B. Subjective belief
            # --------------------------------------------

            agree = int(
                ai_rec[t] == h_rec[t]
            )

            p_subj = fuse_beliefs_logodds(
                np.array([ai_prob[t]]),
                np.array([h_prob[t]]),
                np.array([user.trust_T]),
                np.array([agree]),
                lambda_pos=cfg.lambda_pos,
                lambda_neg=cfg.lambda_neg,
            )[0]

            # --------------------------------------------
            # C. Utility
            # --------------------------------------------

            V_accept = -expected_cost(
                np.array([ai_rec[t]]),
                np.array([p_subj]),
                np.array([c_fp_arr[t]]),
                np.array([c_fn_arr[t]]),
            )[0]

            V_override = -expected_cost(
                np.array([h_rec[t]]),
                np.array([p_subj]),
                np.array([c_fp_arr[t]]),
                np.array([c_fn_arr[t]]),
            )[0] - effort[t]

            utility_term = (
                V_override - V_accept
            )

            # --------------------------------------------
            # D. Institutional incentives
            # --------------------------------------------

            incentive_term = blame_penalty_accept(
                p_ai=np.array([ai_prob[t]]),
                ai_rec=np.array([ai_rec[t]]),
                audit=np.array([audit_true[t]]),
                policy_strict=np.array([policy_true[t]]),
                stake=np.array([stake[t]]),
                blame_scale=cfg.blame_scale,
            )[0]

            # --------------------------------------------
            # E. Belief contribution
            # --------------------------------------------

            belief_term = (
                -cfg.w_proc_now
                * proc_score[t]
                * (
                    1 - cfg.gamma_stress
                    * time_pressure[t]
                )
                if xai_timing[t] == 1
                else 0.0
            )

            # --------------------------------------------
            # F. Override decision score
            # --------------------------------------------

            delta = (
                cfg.w_B * belief_term
                + cfg.w_U * utility_term
                + cfg.w_I * incentive_term
            )

            delta_latent[t] = delta

            utility_latent[t] = utility_term
            incentive_latent[t] = incentive_term
            belief_latent[t] = belief_term

            # --------------------------------------------
            # G. Outcome feedback for future decisions
            # --------------------------------------------

            # The current decision has already been scored.
            # Correctness is revealed only afterward.

            user.trust_T = performance_update(
                user.trust_T,
                ai_correct=bool(
                    ai_rec[t] == y_true[t]
                ),
                stake=float(stake[t]),
                alpha_pos=cfg.alpha_pos,
                alpha_neg=cfg.alpha_neg,
                eta_risk=cfg.eta_risk,
            )

        # ------------------------------------------------
        # 8. Override probability and action execution
        # ------------------------------------------------

        p_override_true, intercept = (
            self._override_probabilities(
                delta_latent,
                ai_rec,
                h_rec,
            )
        )

        override = rng.binomial(
            1,
            p_override_true,
            size=N,
        ).astype(int)

        # Executed action
        y_final = np.where(
            override == 1,
            h_rec,
            ai_rec,
        ).astype(int)

        # An override is effective only if the action changes.
        effective_override = (
            y_final != ai_rec
        ).astype(int)

        # Consistency guarantee
        if not np.array_equal(
            override,
            effective_override,
        ):
            raise ValueError(
                "Override/action inconsistency detected. "
                "Fix: Inspect human recommendations and "
                "override eligibility in generator.py."
            )

        # ------------------------------------------------
        # 9. Apply observability mask
        # ------------------------------------------------

        audit_obs, policy_obs = apply_observability(
            cfg,
            rng,
            audit_true.astype(float),
            policy_true.astype(float),
        )

        # ------------------------------------------------
        # 10. Assemble complete simulation dataset
        # ------------------------------------------------

        df = pd.DataFrame({
            "t": np.arange(N, dtype=int),
            "user_id": user_id.astype(int),

            # Context
            "stake": stake,
            "time_pressure": time_pressure,
            "policy_strict": policy_obs,
            "audit": audit_obs,

            # Ground truth
            "y_true": y_true.astype(int),

            # AI
            "ai_prob": ai_prob,
            "ai_rec": ai_rec.astype(int),

            # Explainability
            "xai_present": xai_present,
            "xai_timing": xai_timing,
            "xai_detail": xai_detail,
            "xai_specificity": xai_specificity,
            "xai_consistency": xai_consistency,
            "xai_faithfulness": xai_faithfulness,

            # Purpose
            "purpose_align": purpose_align,

            # Override behavior
            "override": override,
            "p_override_true": p_override_true,

            # Final decision
            "y_final": y_final,
            "effective_override": effective_override,
        })

        # ------------------------------------------------
        # 11. Optional internal states
        # ------------------------------------------------

        if return_latents and not hover_mode:

            df["human_prob_latent"] = h_prob
            df["human_rec_latent"] = h_rec

            df["trust_T_latent"] = trust_T_latent
            df["delta_latent"] = delta_latent

            df["proc_score_latent"] = proc_score

            df["utility_term_latent"] = utility_latent
            df["incentive_term_latent"] = incentive_latent
            df["belief_term_latent"] = belief_latent

            df["intercept_b"] = float(intercept)

        # ------------------------------------------------
        # 12. Prediction-safe output
        # ------------------------------------------------

        if hover_mode:

            allowed = set(
                RAW_OBSERVABLE_COLS + [TARGET_COL]
            )

            df = df[
                [col for col in df.columns if col in allowed]
            ].copy()

        # ------------------------------------------------
        # 13. Metadata
        # ------------------------------------------------

        for key, value in asdict(cfg).items():
            df.attrs[f"cfg.{key}"] = value

        df.attrs["cfg.intercept_b"] = float(intercept)

        df.attrs["override_rate"] = float(
            override.mean()
        )

        df.attrs["disagreement_rate"] = float(
            (ai_rec != h_rec).mean()
        )

        df.attrs["effective_override_rate"] = float(
            effective_override.mean()
        )

        return df
