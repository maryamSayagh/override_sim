import numpy as np
import pandas as pd
from scipy.stats import norm

# Calibration settings
TARGET_OVERRIDE_RATE = 0.25
MIN_DISAGREEMENT = 0.30
MAX_DISAGREEMENT = 0.45

N = 100_000
SEED = 42

MU_AI_VALUES = np.arange(0.6, 1.61, 0.1)
MU_H_VALUES = np.arange(0.2, 1.21, 0.1)


def calibrate_signals():
    rng = np.random.default_rng(SEED)

    # Same true binary outcomes for all parameter pairs
    y_true = rng.binomial(1, 0.5, size=N)
    s = 2 * y_true - 1

    # Independent noise for AI and human
    noise_ai = rng.normal(0, 1, N)
    noise_h = rng.normal(0, 1, N)

    results = []

    for mu_ai in MU_AI_VALUES:
        for mu_h in MU_H_VALUES:
            # Assumes ctx = 0 during preliminary calibration
            z_ai = mu_ai * s + noise_ai
            z_h = mu_h * s + noise_h

            # sigmoid(z) >= 0.5 is equivalent to z >= 0
            ai_rec = (z_ai >= 0).astype(int)
            h_rec = (z_h >= 0).astype(int)

            ai_accuracy = np.mean(ai_rec == y_true)
            human_accuracy = np.mean(h_rec == y_true)
            disagreement = np.mean(ai_rec != h_rec)

            feasible = disagreement >= TARGET_OVERRIDE_RATE

            conditional_override_needed = (
                TARGET_OVERRIDE_RATE / disagreement
                if disagreement > 0
                else np.nan
            )

            results.append({
                "mu_ai": round(float(mu_ai), 2),
                "mu_h": round(float(mu_h), 2),
                "ai_accuracy": round(float(ai_accuracy), 4),
                "human_accuracy": round(float(human_accuracy), 4),
                "disagreement": round(float(disagreement), 4),
                "override_given_disagreement": round(
                    float(conditional_override_needed), 4
                ),
                "feasible_25pct": feasible,
                "preferred_range": (
                    MIN_DISAGREEMENT
                    <= disagreement
                    <= MAX_DISAGREEMENT
                ),
            })

    df = pd.DataFrame(results)

    df = df.sort_values(
        by=["preferred_range", "disagreement"],
        ascending=[False, True],
    )

    output = "signal_calibration.csv"
    df.to_csv(output, index=False)

    print(f"\nSaved {len(df)} configurations to {output}")

    print("\nConfigurations supporting the preferred disagreement range:")
    selected = df[
        df["preferred_range"]
        & (df["ai_accuracy"] >= df["human_accuracy"])
    ]

    print(selected.head(20).to_string(index=False))


if __name__ == "__main__":
    calibrate_signals()
    