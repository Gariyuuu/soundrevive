"""Reference-fidelity metrics (brief §13). Multiple independent metrics are always reported
together — never rank on one alone."""
from __future__ import annotations

import numpy as np


def _trim_to_same_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant SDR in dB (Le Roux et al. 2019 definition)."""
    estimate, reference = _trim_to_same_length(estimate, reference)
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    ref_energy = np.sum(reference ** 2) + 1e-9
    proj = np.sum(estimate * reference) / ref_energy * reference
    noise = estimate - proj
    ratio = (np.sum(proj ** 2) + 1e-9) / (np.sum(noise ** 2) + 1e-9)
    return float(10 * np.log10(ratio))


def sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate, reference = _trim_to_same_length(estimate, reference)
    noise = estimate - reference
    ratio = (np.sum(reference ** 2) + 1e-9) / (np.sum(noise ** 2) + 1e-9)
    return float(10 * np.log10(ratio))


def snr(audio: np.ndarray, reference: np.ndarray) -> float:
    """Plain SNR of `audio` against `reference` treated as the clean signal."""
    return sdr(audio, reference)


def snr_improvement(degraded: np.ndarray, restored: np.ndarray, reference: np.ndarray) -> float:
    return snr(restored, reference) - snr(degraded, reference)


def log_spectral_distance(estimate: np.ndarray, reference: np.ndarray, n_fft: int = 512, hop: int = 128) -> float:
    from soundrevive.stft_utils import stft

    estimate, reference = _trim_to_same_length(estimate, reference)
    spec_e = np.abs(stft(estimate, n_fft, hop)) + 1e-8
    spec_r = np.abs(stft(reference, n_fft, hop)) + 1e-8
    n = min(spec_e.shape[1], spec_r.shape[1])
    log_diff = 20 * (np.log10(spec_e[:, :n]) - np.log10(spec_r[:, :n]))
    return float(np.sqrt(np.mean(log_diff ** 2)))
