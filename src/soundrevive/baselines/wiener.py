"""Classical Wiener filtering baseline (brief §10) — a priori SNR estimated via decision-
directed approach (Ephraim-Malah style), noise PSD from the lowest-energy frame percentile."""
from __future__ import annotations

import numpy as np

from soundrevive.stft_utils import istft, stft


def wiener_filter(audio: np.ndarray, sr: int, n_fft: int = 512, hop: int = 128,
                   noise_percentile: float = 10.0, smoothing: float = 0.98) -> np.ndarray:
    spec = stft(audio, n_fft, hop)
    power = np.abs(spec) ** 2
    noise_power = np.percentile(power, noise_percentile, axis=1, keepdims=True)

    n_freq, n_frames = power.shape
    gain = np.ones_like(power)
    prev_gain_power = np.zeros(n_freq)
    for t in range(n_frames):
        post_snr = np.maximum(power[:, t] / (noise_power[:, 0] + 1e-12) - 1, 0)
        a_priori = smoothing * prev_gain_power / (noise_power[:, 0] + 1e-12) + (1 - smoothing) * post_snr
        g = a_priori / (a_priori + 1)
        gain[:, t] = g
        prev_gain_power = g * power[:, t]

    out_spec = spec * gain
    return istft(out_spec, n_fft, hop, length=len(audio))
