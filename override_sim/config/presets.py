from __future__ import annotations
from .schemas import SimConfig


def phase1() -> SimConfig:
    # Start with defaults; you can tweak later.
    return SimConfig()
