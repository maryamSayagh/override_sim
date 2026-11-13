""" 
initial method we'll use to check whether the dataset generated in fact follows the probabilities and assumptions we expect from input knowledge. 
these expectation are basically what the generator used to create each row. 

simulator generates: [features, [override , y_override_proba]]
model: fit on [features, [override,y_override_proba]]
       test and validate on [features], returns [features, y_proba]
calibration: measures whether y_override_proba ~= y_proba


"""
"""
Initial method to evaluate whether the generated dataset
preserves the probabilistic patterns and assumptions specified in the simulator.

Simulator generates:
    [features, [override, y_override_proba]]

Model:
    Train on [features] with override as the target label. 
    test and validate on [features], returns [features, y_proba]
    and y_proba is probability of override=0 or 1 for each row 

Evaluation:
    Probability recovery:
        Compare y_proba with y_override_proba.

    Calibration:
        "Among all rows where y_proba ≈ 0.8, approximately 80% should have override = 1 for the model to be well calibrated."
        measured by: 
            ECE: A number between 0 and 1; closer to 0 means better calibration.
            Reliability curve: A graph showing how closely predicted probabilities match observed frequencies.

    Interpretation:
        Good probability recovery suggests the intended
        probabilistic relationships are learnable from
        observable features.
"""




import numpy as np
from sklearn.calibration import calibration_curve

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y = np.asarray(y_true)
    p = np.asarray(y_prob)

    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(
        np.digitize(p, edges[1:-1]), 0, n_bins - 1
    )

    ece = 0.0

    for i in range(n_bins):
        mask = bins == i

        if mask.any():
            confidence = p[mask].mean()
            observed = y[mask].mean()

            ece += mask.mean() * abs(
                confidence - observed
            )

    return float(ece)


def reliability_curve(y_true, y_prob, n_bins=10):
    observed, predicted = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy="uniform",
    )

    return {
        "predicted": predicted.tolist(),
        "observed": observed.tolist(),
    }
