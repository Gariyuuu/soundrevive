"""Classical median-filter click/pop detector + interpolation (brief §10, §37). Detects
short impulses as large deviations from a local median, then linearly interpolates across
flagged spans from their (unflagged) neighbors.

The detection threshold is LOCAL (a second, wider median filter over |residual|), not a
single global MAD over the whole clip: a global threshold was tried first and made things
worse on real speech (tests/test_baselines.py caught this — declicking increased error
relative to the clicked input, because loud passages produce residual spikes from ordinary
fast speech transients, like plosives, that a global threshold can't distinguish from actual
clicks). A local noise-floor estimate adapts to quiet vs. loud passages instead.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import medfilt


def median_declick(audio: np.ndarray, sr: int, window_ms: float = 1.0, local_window_ms: float = 20.0,
                    threshold_k: float = 18.0) -> np.ndarray:
    def odd_window(ms: float) -> int:
        w = max(3, int(ms / 1000.0 * sr))
        return w + 1 if w % 2 == 0 else w

    window = odd_window(window_ms)
    local_window = odd_window(local_window_ms)

    baseline = medfilt(audio, kernel_size=window)
    residual = audio - baseline
    local_scale = medfilt(np.abs(residual), kernel_size=local_window) + 1e-6
    flagged = np.abs(residual) > threshold_k * local_scale * 1.4826

    out = audio.copy()
    idx = np.arange(len(audio))
    if flagged.any() and not flagged.all():
        out[flagged] = np.interp(idx[flagged], idx[~flagged], audio[~flagged])
    return out.astype(np.float32)
