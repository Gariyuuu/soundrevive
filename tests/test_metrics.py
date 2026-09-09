import numpy as np

from soundrevive.metrics import (
    align,
    clipping_fraction,
    log_spectral_distance,
    si_sdr,
    snr_improvement,
    word_error_rate,
)


def _tone(sr=16000, dur_s=1.0, freq=220.0):
    t = np.arange(int(sr * dur_s)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_si_sdr_perfect_reconstruction_is_very_high():
    ref = _tone()
    assert si_sdr(ref, ref) > 100


def test_si_sdr_scale_invariance():
    ref = _tone()
    scaled = ref * 3.7
    assert si_sdr(scaled, ref) > 100  # scale-invariant: pure rescale is "perfect"


def test_si_sdr_degrades_with_added_noise():
    rng = np.random.default_rng(0)
    ref = _tone()
    noisy = ref + 0.5 * rng.standard_normal(len(ref)).astype(np.float32)
    assert si_sdr(noisy, ref) < si_sdr(ref, ref) - 10


def test_snr_improvement_positive_when_restoration_helps():
    rng = np.random.default_rng(0)
    ref = _tone()
    noise = 0.3 * rng.standard_normal(len(ref)).astype(np.float32)
    degraded = ref + noise
    restored = ref + 0.5 * noise  # halfway "denoised"
    assert snr_improvement(degraded, restored, ref) > 0


def test_log_spectral_distance_zero_for_identical_signal():
    ref = _tone()
    assert log_spectral_distance(ref, ref) < 1e-3


def test_alignment_recovers_known_integer_shift():
    ref = _tone(dur_s=2.0)
    shift = 37
    shifted = np.concatenate([np.zeros(shift, dtype=np.float32), ref])[: len(ref)]
    aligned_est, aligned_ref, lag = align(shifted, ref, max_lag=200)
    assert lag == shift
    assert si_sdr(aligned_est, aligned_ref) > 100


def test_clipping_fraction_detects_clipped_samples():
    audio = np.array([0.1, 0.5, 1.0, -1.0, 0.2], dtype=np.float32)
    assert clipping_fraction(audio) == 2 / 5


def test_clipping_fraction_zero_for_clean_signal():
    ref = _tone() * 0.5
    assert clipping_fraction(ref) == 0.0


def test_word_error_rate_normalizes_case_and_punctuation():
    # Real regression: comparing a Whisper hypothesis against a LibriSpeech-style
    # ground-truth transcript (all-caps, no punctuation, spelled-out abbreviations)
    # scored WER=1.0 on a perfect transcription before normalization was added.
    reference = "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES"
    hypothesis = "Mr. Quilter is the apostle of the middle classes."
    assert word_error_rate(hypothesis, reference) == 0.0
