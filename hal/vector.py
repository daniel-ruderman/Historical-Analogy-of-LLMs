"""Vector helpers for the retrieval methods (numpy when available)."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

try:  # pragma: no cover - depends on the environment
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


def dot(x: Sequence[float], y: Sequence[float]) -> float:
    """Inner product -- what the original repository uses as its similarity."""
    if _np is not None:
        return float(_np.dot(_np.asarray(x, dtype=float), _np.asarray(y, dtype=float)))
    return float(sum(a * b for a, b in zip(x, y)))


def norm(x: Sequence[float]) -> float:
    if _np is not None:
        return float(_np.linalg.norm(_np.asarray(x, dtype=float)))
    return math.sqrt(sum(a * a for a in x))


def l2_normalize(x: Sequence[float]) -> List[float]:
    n = norm(x)
    if n == 0:
        return list(map(float, x))
    return [float(a) / n for a in x]


def cosine_similarity(x: Sequence[float], y: Sequence[float]) -> float:
    denom = norm(x) * norm(y)
    if denom == 0:
        return 0.0
    return dot(x, y) / denom


def rank_by_similarity(query: Sequence[float],
                       items: Iterable[Tuple[object, Sequence[float]]]
                       ) -> List[Tuple[object, float]]:
    """Sort ``(item, vector)`` pairs by descending cosine similarity to ``query``."""
    scored = [(item, cosine_similarity(query, vector)) for item, vector in items]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
