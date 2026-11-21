"""4-D semantic vector codec (paper Sec. II-B, Eq. 6).

The semantic payload is exactly 16 bytes: four little-endian float32 confidence
scores [Normal Flow, Congestion, Accident, Invalid Scene] that sum to 1.
Metadata (RSU id, sequence number, send time) travels in HTTP headers so the
payload itself stays 16 bytes, as in the paper.
"""
from __future__ import annotations

import struct

import numpy as np

CLASSES = ["normal_flow", "congestion", "accident", "invalid_scene"]
CLASS_LABELS = ["Normal Flow", "Congestion", "Accident", "Invalid Scene"]
PAYLOAD_SIZE = 16
_FMT = "<4f"


def encode(m) -> bytes:
    v = np.asarray(m, dtype=np.float32).reshape(-1)
    if v.shape != (4,):
        raise ValueError(f"semantic vector must have 4 elements, got {v.shape}")
    return struct.pack(_FMT, *v.tolist())


def decode(b: bytes) -> np.ndarray:
    if len(b) != PAYLOAD_SIZE:
        raise ValueError(f"payload must be {PAYLOAD_SIZE} bytes, got {len(b)}")
    v = np.array(struct.unpack(_FMT, b), dtype=np.float32)
    if not np.all(np.isfinite(v)) or np.any(v < 0):
        raise ValueError("payload contains invalid confidence values")
    return v


def top_class(m) -> tuple[str, float]:
    v = np.asarray(m)
    i = int(np.argmax(v))
    return CLASSES[i], float(v[i])
