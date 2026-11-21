"""Confidence-aware Partner Verification (paper Algorithm 1, Eq. 7).

Pure numpy, no model dependency, so it is unit-testable and reusable.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# Paper Sec. IV: tau_s = 60%, tau_delta = 30%
TAU_S = 0.60
TAU_DELTA = 0.30

# A partner takes the image and the two candidate class indices and returns one of them.
Partner = Callable[[object, tuple[int, int]], int]


def needs_verification(m, tau_s: float = TAU_S, tau_delta: float = TAU_DELTA) -> bool:
    v = np.asarray(m, dtype=np.float64)
    order = np.argsort(v)[::-1]
    s1, s2 = v[order[0]], v[order[1]]
    return not (s1 >= tau_s or (s1 - s2) >= tau_delta)


def verify(
    m,
    image,
    partners: Sequence[Partner],
    tau_s: float = TAU_S,
    tau_delta: float = TAU_DELTA,
) -> tuple[np.ndarray, dict]:
    """Return (M*, info). M* keeps the same 4-D shape/size as M."""
    v = np.asarray(m, dtype=np.float32).copy()
    order = np.argsort(v)[::-1]
    i1, i2 = int(order[0]), int(order[1])
    info = {"triggered": False, "swapped": False, "votes": [], "top1": i1, "top2": i2}

    if not needs_verification(v, tau_s, tau_delta):
        return v, info

    info["triggered"] = True
    votes = [int(p(image, (i1, i2))) for p in partners]
    info["votes"] = votes
    if len(votes) >= 2 and all(x == i2 for x in votes):
        v[i1], v[i2] = v[i2], v[i1]
        info["swapped"] = True
    return v, info
