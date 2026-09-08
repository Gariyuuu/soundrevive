"""Degradation primitives. Every function is a pure, deterministic map
(audio, sr, seed, **params) -> audio_out, given the same inputs it always returns bit-identical
output (verified by tests/test_degradation_determinism.py). All primitives preserve array
length so composed chains never need to reconcile differing output lengths.

Historical motivation for parameter ranges: see research/restoration_taxonomy.md. These are
documented engineering approximations, not literature-verified constants, until cross-checked
against research/literature_matrix.csv.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def bandlimit(audio: np.ndarray, sr: int, seed: int, low_hz: float = 100.0, high_hz: float = 5000.0, order: int = 4) -> np.ndarray:
    nyq = sr / 2.0
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def surface_noise(audio: np.ndarray, sr: int, seed: int, snr_db: float = 20.0, color_alpha: float = 1.0) -> np.ndarray:
    """Colored (1/f^alpha) additive noise, scaled to hit the target broadband SNR."""
    rng = _rng(seed)
    n = len(audio)
    white = rng.standard_normal(n).astype(np.float32)
    if color_alpha > 0:
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        freqs[0] = freqs[1] if n > 1 else 1.0
        shaping = 1.0 / (freqs ** (color_alpha / 2.0))
        shaping = shaping / np.sqrt(np.mean(shaping ** 2))
        colored = np.fft.irfft(spec * shaping, n=n).astype(np.float32)
    else:
        colored = white
    sig_rms = _rms(audio)
    target_noise_rms = sig_rms / (10 ** (snr_db / 20.0))
    noise_rms = _rms(colored)
    if noise_rms > 1e-9:
        colored = colored * (target_noise_rms / noise_rms)
    return (audio + colored).astype(np.float32)


def hum(audio: np.ndarray, sr: int, seed: int, freq_hz: float = 60.0, n_harmonics: int = 3, amp_db: float = -30.0) -> np.ndarray:
    n = len(audio)
    t = np.arange(n) / sr
    sig_rms = _rms(audio)
    amp = sig_rms * (10 ** (amp_db / 20.0))
    out = np.zeros(n, dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        out += (1.0 / k) * np.sin(2 * np.pi * freq_hz * k * t)
    out_rms = _rms(out) if out.any() else 1.0
    out = out * (amp / max(out_rms, 1e-9))
    return (audio + out.astype(np.float32)).astype(np.float32)


def clicks_pops(audio: np.ndarray, sr: int, seed: int, rate_per_sec: float = 2.0, amplitude_db: float = -6.0, width_ms: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (degraded_audio, click_mask) — mask=1 where a click was injected, for
    declick-evaluation ground truth (brief §37)."""
    rng = _rng(seed)
    n = len(audio)
    duration_s = n / sr
    n_clicks = rng.poisson(rate_per_sec * duration_s)
    out = audio.copy()
    mask = np.zeros(n, dtype=np.float32)
    width = max(1, int(width_ms / 1000.0 * sr))
    sig_peak = float(np.max(np.abs(audio)) + 1e-6)
    amp = sig_peak * (10 ** (amplitude_db / 20.0))
    for _ in range(n_clicks):
        pos = int(rng.integers(0, max(1, n - width)))
        env = np.hanning(width * 2)[:width] if width > 1 else np.ones(1)
        polarity = 1.0 if rng.random() > 0.5 else -1.0
        out[pos:pos + width] += (polarity * amp * env[: n - pos]).astype(np.float32)
        mask[pos:pos + width] = 1.0
    return out.astype(np.float32), mask


def crackle(audio: np.ndarray, sr: int, seed: int, density: float = 20.0, amplitude_db: float = -20.0) -> tuple[np.ndarray, np.ndarray]:
    return clicks_pops(audio, sr, seed, rate_per_sec=density, amplitude_db=amplitude_db, width_ms=0.5)


def wow_flutter(audio: np.ndarray, sr: int, seed: int, wow_hz: float = 1.0, wow_depth: float = 0.002, flutter_hz: float = 10.0, flutter_depth: float = 0.0005) -> np.ndarray:
    n = len(audio)
    t = np.arange(n) / sr
    warp = wow_depth * np.sin(2 * np.pi * wow_hz * t) + flutter_depth * np.sin(2 * np.pi * flutter_hz * t)
    read_idx = np.arange(n) + warp * sr
    read_idx = np.clip(read_idx, 0, n - 1)
    return np.interp(read_idx, np.arange(n), audio).astype(np.float32)


def nonlinear_distortion(audio: np.ndarray, sr: int, seed: int, drive_db: float = 6.0, curve: str = "tanh") -> np.ndarray:
    driven = audio * (10 ** (drive_db / 20.0))
    if curve == "hard_clip":
        out = np.clip(driven, -1.0, 1.0)
    else:
        out = np.tanh(driven)
    peak = float(np.max(np.abs(audio)) + 1e-9)
    out_peak = float(np.max(np.abs(out)) + 1e-9)
    return (out * (peak / out_peak)).astype(np.float32)


def quantization(audio: np.ndarray, sr: int, seed: int, bits: int = 8) -> np.ndarray:
    levels = 2 ** bits
    q = np.round(audio * (levels / 2 - 1)) / (levels / 2 - 1)
    return q.astype(np.float32)


def dropout(audio: np.ndarray, sr: int, seed: int, n_dropouts: int = 3, duration_ms_range: tuple[float, float] = (20.0, 120.0)) -> tuple[np.ndarray, np.ndarray]:
    """Returns (degraded_audio, dropout_mask) — mask=1 where signal was zeroed, the exact
    ground-truth region for dropout-inpainting evaluation (brief §38)."""
    rng = _rng(seed)
    n = len(audio)
    out = audio.copy()
    mask = np.zeros(n, dtype=np.float32)
    for _ in range(n_dropouts):
        dur_ms = rng.uniform(*duration_ms_range)
        width = max(1, int(dur_ms / 1000.0 * sr))
        pos = int(rng.integers(0, max(1, n - width)))
        fade = min(64, width // 4)
        env = np.ones(width, dtype=np.float32)
        if fade > 0:
            env[:fade] = np.linspace(1, 0, fade)
            env[-fade:] = np.linspace(0, 1, fade)
        out[pos:pos + width] *= (1.0 - env[: n - pos])
        mask[pos:pos + width] = 1.0
    return out.astype(np.float32), mask


def reverb(audio: np.ndarray, sr: int, seed: int, rt60_s: float = 0.5, wet_level: float = 0.3) -> np.ndarray:
    """Simplified synthetic-IR reverb (exponentially decaying filtered noise). Documented as
    an approximation, not a measured room impulse response — see restoration_taxonomy.md."""
    rng = _rng(seed)
    ir_len = max(1, int(rt60_s * sr))
    ir = rng.standard_normal(ir_len).astype(np.float32)
    decay = np.exp(-6.9078 * np.arange(ir_len) / max(ir_len, 1)).astype(np.float32)  # -60 dB at rt60
    ir = ir * decay
    ir = ir / (np.sqrt(np.sum(ir ** 2)) + 1e-9)
    wet = signal.fftconvolve(audio, ir, mode="full")[: len(audio)].astype(np.float32)
    return ((1 - wet_level) * audio + wet_level * wet).astype(np.float32)


PRIMITIVES = {
    "bandlimit": bandlimit,
    "surface_noise": surface_noise,
    "hum": hum,
    "clicks_pops": clicks_pops,
    "crackle": crackle,
    "wow_flutter": wow_flutter,
    "nonlinear_distortion": nonlinear_distortion,
    "quantization": quantization,
    "dropout": dropout,
    "reverb": reverb,
}

# primitives that return (audio, mask) instead of just audio
MASK_PRIMITIVES = {"clicks_pops", "crackle", "dropout"}
