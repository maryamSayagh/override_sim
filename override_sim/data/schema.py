# Columns allowed to be used at prediction time (observable at recommendation time)
OBSERVABLE_COLS = [
    "t",
    "user_id",
    "stake",
    "time_pressure",
    "policy_strict",
    "audit",
    "ai_prob",
    "ai_rec",
]

# Columns explicitly forbidden for training the override predictor (leakage)
FORBIDDEN_COLS = [
    "y_true",
    "confusion_type",
    "p_override_true",
    "human_prob_latent",
    "human_rec_latent",
    "trust_T_latent",
    "delta_latent",
    "intercept_b",
]
