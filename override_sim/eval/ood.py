from sklearn.base import clone
from .metrics import compute_metrics

def evaluate_transfer(
    model,
    worlds,
    features,
    target="override",
):
    results = []

    for train_name, train_df in worlds.items():
        for test_name, test_df in worlds.items():

            estimator = clone(model)

            estimator.fit(
                train_df[features],
                train_df[target],
            )

            probabilities = estimator.predict_proba(
                test_df[features]
            )[:, 1]

            metrics = compute_metrics(
                test_df[target],
                probabilities,
            )

            results.append({
                "train_world": train_name,
                "test_world": test_name,
                **metrics,
            })

    return results
