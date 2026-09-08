"""Time alignment before sample-level comparison (brief §18). Restoration models (STFT/ISTFT
framing, resampling) can introduce a constant group delay; this finds and corrects it via
FFT cross-correlation before any fidelity metric touches the signal."""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate


def find_lag(estimate: np.ndarray, reference: np.ndarray, max_lag: int | None = None) -> int:
    """Positive lag means `estimate` is delayed relative to `reference` (must be shifted
    earlier to align)."""
    n = min(len(estimate), len(reference))
    e = estimate[:n] - estimate[:n].mean()
    r = reference[:n] - reference[:n].mean()
    corr = correlate(e, r, mode="full", method="fft")
    lags = np.arange(-n + 1, n)
    if max_lag is not None:
        keep = np.abs(lags) <= max_lag
        corr, lags = corr[keep], lags[keep]
    return int(lags[np.argmax(corr)])


def align(estimate: np.ndarray, reference: np.ndarray, max_lag: int = 4000) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (aligned_estimate, aligned_reference, lag_used), both trimmed to equal,
    overlapping length."""
    lag = find_lag(estimate, reference, max_lag=max_lag)
    if lag > 0:
        estimate = estimate[lag:]
    elif lag < 0:
        reference = reference[-lag:]
    n = min(len(estimate), len(reference))
    return estimate[:n], reference[:n], lag
