""" initial method we'll use to check whether the dataset generated in fact follows the probabilities and assumptions we expect from input knowledge. 
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
