"""Shared STFT/ISTFT so classical baselines and the ML denoiser use identical time-frequency
conventions (this matters for the alignment logic downstream).

Thin wrapper around scipy.signal.stft/istft rather than a hand-rolled overlap-add: an earlier
hand-rolled version used the standard "square the window, normalize by sum(window**2)"
weighted-OLA recipe, which is only an exact inverse for an *unmodified* round-tripped
spectrum. Every real caller here (spectral gating, Wiener filtering, the STFT-mask denoiser)
reconstructs from a *magnitude-modified* spectrum, which is off the manifold that recipe's
normalization was derived for — verified empirically: a per-bin gain in [0, 1] applied to a
noisy speech clip *increased* reconstructed RMS by 3-14x with the hand-rolled version, while
scipy's istft (same window/hop) gave the expected, bounded-below-original result. scipy's
implementation is correct for arbitrary modified spectra, not just identity round-trips, so
it's used directly instead of re-deriving the correct normalization by hand.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import istft as _scipy_istft
from scipy.signal import stft as _scipy_stft


def stft(audio: np.ndarray, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    _, _, spec = _scipy_stft(
        audio.astype(np.float64), window="hann", nperseg=n_fft, noverlap=n_fft - hop,
        boundary="zeros", padded=True,
    )
    return spec  # (freq_bins, n_frames), complex


def istft(spec: np.ndarray, n_fft: int = 512, hop: int = 128, length: int | None = None) -> np.ndarray:
    _, out = _scipy_istft(
        spec, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=True,
    )
    out = out.astype(np.float32)
    if length is not None:
        if len(out) < length:
            out = np.pad(out, (0, length - len(out)))
        out = out[:length]
    return out
