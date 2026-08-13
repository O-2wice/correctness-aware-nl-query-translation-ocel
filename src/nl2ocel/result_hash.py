"""Stable denotation hashing for benchmark result dataframes.

The evaluator compares generated answers with gold SQL answers by hashing the
returned values. DuckDB floating aggregates such as STDDEV can differ in the
last binary-floating digit across equivalent execution plans, so float cells
are rounded before hashing. Non-float values keep their normal Python list
representation, preserving the original value-only, order-independent scheme.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real
from typing import Any


FLOAT_DECIMALS = 6


def _normalise_value(value: Any) -> Any:
    """Return a stable scalar representation for one dataframe cell."""
    if isinstance(value, Real) and not isinstance(value, (Integral, bool)):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return round(value, FLOAT_DECIMALS)
    return value


def hash_dataframe(df) -> str:
    """Value-only SHA-256 hash, independent of row order and column aliases."""
    rows = []
    for row in df.itertuples(index=False, name=None):
        rows.append(str([_normalise_value(v) for v in row]))
    return hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()
