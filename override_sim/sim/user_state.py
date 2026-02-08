from __future__ import annotations
from dataclasses import dataclass


def clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@dataclass
class UserState:
    trust_T: float

    def update(self, ai_correct: bool, alpha_pos: float, alpha_neg: float) -> None:
        # Asymmetric trust learning: errors reduce trust more than successes increase it.
        if ai_correct:
            self.trust_T = clip01(self.trust_T + alpha_pos * (1.0 - self.trust_T))
        else:
            self.trust_T = clip01(self.trust_T - alpha_neg * self.trust_T)
