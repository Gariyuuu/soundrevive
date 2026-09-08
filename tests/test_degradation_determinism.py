import numpy as np

from soundrevive.degradation import TIERS, apply_chain, sample_chain_config


def _tone(sr=16000, dur_s=1.0, freq=220.0):
    t = np.arange(int(sr * dur_s)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_same_seed_gives_byte_identical_output():
    audio = _tone()
    cfg = sample_chain_config("MODERATE", seed=7, sample_rate=16000)
    out1, mask1, _ = apply_chain(audio, cfg)
    out2, mask2, _ = apply_chain(audio, cfg)
    assert np.array_equal(out1, out2)
    assert np.array_equal(mask1, mask2)


def test_different_seed_gives_different_output():
    audio = _tone()
    cfg_a = sample_chain_config("MODERATE", seed=1, sample_rate=16000)
    cfg_b = sample_chain_config("MODERATE", seed=2, sample_rate=16000)
    out_a, _, _ = apply_chain(audio, cfg_a)
    out_b, _, _ = apply_chain(audio, cfg_b)
    assert not np.array_equal(out_a, out_b)


def test_output_length_preserved_for_all_tiers():
    audio = _tone(dur_s=2.0)
    for tier in TIERS:
        cfg = sample_chain_config(tier, seed=3, sample_rate=16000)
        out, mask, _ = apply_chain(audio, cfg)
        assert len(out) == len(audio)
        assert len(mask) == len(audio)


def test_resolved_config_is_json_serializable_and_reproduces_output():
    import json

    audio = _tone()
    cfg = sample_chain_config("SEVERE", seed=11, sample_rate=16000)
    out1, _, resolved = apply_chain(audio, cfg)
    json.dumps(resolved)  # must not raise
    out2, _, _ = apply_chain(audio, resolved)
    assert np.array_equal(out1, out2)


def test_severity_ordering_reduces_snr_on_average():
    """Sanity check on tier definitions: EXTREME should be noisier than LIGHT for the same
    seed and source, on average — not a strict guarantee for every single seed given random
    click placement, so this checks total energy of the injected damage, not a metric like
    SI-SDR (which needs a separate, non-circular test)."""
    audio = _tone(dur_s=2.0)
    light, _, _ = apply_chain(audio, sample_chain_config("LIGHT", seed=5, sample_rate=16000))
    extreme, _, _ = apply_chain(audio, sample_chain_config("EXTREME", seed=5, sample_rate=16000))
    light_residual_energy = np.sum((light - audio) ** 2)
    extreme_residual_energy = np.sum((extreme - audio) ** 2)
    assert extreme_residual_energy > light_residual_energy
