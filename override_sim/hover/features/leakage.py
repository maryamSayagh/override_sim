from __future__ import annotations
from ..data.schema import FORBIDDEN_COLS


class LeakageGuard:
    def assert_no_forbidden_columns(self, df) -> None:
        bad = [c for c in df.columns if c in set(FORBIDDEN_COLS)]
        if bad:
            raise ValueError(f"LeakageGuard: forbidden columns present: {bad}")
