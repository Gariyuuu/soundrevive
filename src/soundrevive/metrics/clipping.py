"""Clipping check on restoration output (brief §34)."""
from __future__ import annotations

import numpy as np


def clipping_fraction(audio: np.ndarray, threshold: float = 0.999) -> float:
    return float(np.mean(np.abs(audio) >= threshold))
