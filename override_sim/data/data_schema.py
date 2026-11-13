# ============================================================
# 1) RAW OBSERVABLE COLUMNS
# Directly available at recommendation time
# ============================================================
RAW_OBSERVABLE_COLS = [
    # indexing / sequencing
    "t",
    "user_id",

    # context
    "stake",
    "time_pressure",
    "policy_strict",
    "audit",

    # AI output
    "ai_prob",
    "ai_rec",

    # explainability / process
    "xai_present",
    "xai_timing",          # 0=none, 1=pre, 2=post
    "xai_detail",
    "xai_specificity",
    "xai_consistency",
    "xai_faithfulness",

    # purpose / alignment proxy
    "purpose_align",
]

# ============================================================
# 2) DERIVED FEATURES ALLOWED FOR PREDICTION
# Computed from current observable data and/or past-only history
# ============================================================
DERIVED_ALLOWED_COLS = [
    # belief / trust / behavior proxies
    "B_ovr_ewma",
    "B_ovr_rolling10",
    "B_ai_prob_ewma",
    "B_ai_acc_ewma",
    "B_ai_err_rolling20",

    # explainability / process features
    "B_xai_present",
    "B_xai_pre",
    "B_xai_post",
    "B_xai_proc_score",
    "B_xai_faithfulness",
    "B_purpose_align",

    # utility / risk features
    "U_stake",
    "U_risk_pressure",

    # incentives / institutional features
    "I_audit",
    "I_policy",
    "I_audit_policy",
    "I_audit_x_stake",
    "I_audit_x_uncertainty",

    # cross-block interactions
    "BU_uncertainty_x_stake",
    "BI_uncertainty_x_audit",
]

# ============================================================
# 3) DERIVED BUT FORBIDDEN
# Computed variables that leak target logic or use unavailable info
# ============================================================
DERIVED_FORBIDDEN_COLS = [
    "p_override_true",
    "intercept_b",
    "confusion_type",
]

# ============================================================
# 4) LATENT / INTERNAL VARIABLES
# Behavioral variables used by the generator only
# ============================================================
LATENT_COLS = [
    "human_prob_latent",
    "human_rec_latent",
    "trust_T_latent",
    "delta_latent",
    "proc_score_latent",
]

# ============================================================
# 5) POST-OUTCOME / FUTURE-ONLY VARIABLES
# Not available at recommendation time
# ============================================================
POST_OUTCOME_COLS = [
    "y_true",
]

# ============================================================
# 6) FINAL REGISTRIES
# ============================================================
ALLOWED_FEATURE_COLS = RAW_OBSERVABLE_COLS + DERIVED_ALLOWED_COLS

FORBIDDEN_COLS = DERIVED_FORBIDDEN_COLS + LATENT_COLS + POST_OUTCOME_COLS

ALL_KNOWN_COLS = (
    RAW_OBSERVABLE_COLS
    + DERIVED_ALLOWED_COLS
    + DERIVED_FORBIDDEN_COLS
    + LATENT_COLS
    + POST_OUTCOME_COLS
)

TARGET_COL = "override"
