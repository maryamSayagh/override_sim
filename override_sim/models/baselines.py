from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


@dataclass
class LogisticOverrideModel:
    C: float = 1.0
    max_iter: int = 2000

    def __post_init__(self):
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


@dataclass
class GBDTOverrideModel:
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3

    def __post_init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
