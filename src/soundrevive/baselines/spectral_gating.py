"""Classical spectral-gating noise reduction (brief §10). Noise magnitude spectrum is
estimated from the lowest-energy percentile of frames (no assumption of a silent lead-in),
then subtracted per bin with a floor to avoid musical-noise artifacts blowing up."""
from __future__ import annotations

import numpy as np

from soundrevive.stft_utils import istft, stft


def spectral_gate(audio: np.ndarray, sr: int, n_fft: int = 512, hop: int = 128,
                   noise_percentile: float = 10.0, oversubtraction: float = 1.5,
                   floor: float = 0.05) -> np.ndarray:
    spec = stft(audio, n_fft, hop)
    mag = np.abs(spec)
    phase = np.angle(spec)
    noise_mag = np.percentile(mag, noise_percentile, axis=1, keepdims=True)
    gated = mag - oversubtraction * noise_mag
    gated = np.maximum(gated, floor * mag)
    out_spec = gated * np.exp(1j * phase)
    return istft(out_spec, n_fft, hop, length=len(audio))
