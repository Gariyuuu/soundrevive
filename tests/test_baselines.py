import numpy as np

from soundrevive.baselines import (
    median_declick,
    passthrough,
    spectral_gate,
    wiener_filter,
)
from soundrevive.degradation.primitives import clicks_pops, surface_noise
from soundrevive.io_utils import load_wav
from soundrevive.pipeline.manifest import clip_path


def _tone(sr=16000, dur_s=1.0, freq=220.0):
    t = np.arange(int(sr * dur_s)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _speech(dur_s=4.0):
    """Real committed clean-speech clip, not a synthetic tone: a stationary sine's period
    can be comparable to a classical baseline's analysis window (median-filter kernel,
    percentile-noise-floor STFT frame), which is a pathological case these algorithms were
    never meant to handle — speech doesn't have that failure mode, so effectiveness checks
    below use real speech instead of a tone."""
    audio, sr = load_wav(str(clip_path("alice00_16k_clip")))
    n = int(dur_s * sr)
    return audio[:n], sr


def test_passthrough_is_identity():
    audio = _tone()
    assert np.array_equal(passthrough(audio, 16000), audio)
    assert passthrough(audio, 16000) is not audio  # must be a copy, not aliasing


def test_all_baselines_preserve_length():
    audio = _tone()
    assert len(spectral_gate(audio, 16000)) == len(audio)
    assert len(wiener_filter(audio, 16000)) == len(audio)
    assert len(median_declick(audio, 16000)) == len(audio)


def test_spectral_gate_reduces_noise_energy_relative_to_clean_speech():
    clean, sr = _speech()
    noisy = surface_noise(clean, sr, seed=1, snr_db=10.0)
    gated = spectral_gate(noisy, sr)
    noise_before = np.mean((noisy - clean) ** 2)
    noise_after = np.mean((gated - clean) ** 2)
    assert noise_after < noise_before


def test_median_declick_reduces_click_energy():
    clean, sr = _speech()
    clicked, _mask = clicks_pops(clean, sr, seed=1, rate_per_sec=5.0, amplitude_db=0.0)
    declicked = median_declick(clicked, sr)
    err_before = np.mean((clicked - clean) ** 2)
    err_after = np.mean((declicked - clean) ** 2)
    assert err_after < err_before
