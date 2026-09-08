"""Quantitative tier definitions (research/restoration_taxonomy.md §3) and the canonical
"historical-style compound chain" (brief §7) that composes primitives in a documented order.
Sampling a chain for a tier is itself deterministic given a seed."""
from __future__ import annotations

import numpy as np

TIERS = ["LIGHT", "MODERATE", "SEVERE", "EXTREME"]

# Each entry: (param_range_low, param_range_high) sampled uniformly per tier.
# bandlimit_low/bandlimit_high are the two edges of the passband sampled independently —
# NOT a single range the low edge is drawn from (that was a bug: it let the "LIGHT" low edge
# land above the "EXTREME" high edge, sometimes filtering out all the same content regardless
# of tier). Ranges narrow and shift as severity increases, per research/restoration_taxonomy.md §3.
_TIER_PARAMS = {
    "LIGHT": dict(
        bandlimit_low=(80, 200), bandlimit_high=(4500, 6000), snr_db=(25, 35), click_rate=(0.5, 2.0),
        wow_depth=(0.0002, 0.001), hum_db=(-45, -35), drive_db=(0, 1),
    ),
    "MODERATE": dict(
        bandlimit_low=(150, 300), bandlimit_high=(3500, 4500), snr_db=(15, 25), click_rate=(2.0, 6.0),
        wow_depth=(0.001, 0.003), hum_db=(-38, -28), drive_db=(1, 3),
    ),
    "SEVERE": dict(
        bandlimit_low=(250, 400), bandlimit_high=(2800, 3500), snr_db=(5, 15), click_rate=(6.0, 15.0),
        wow_depth=(0.003, 0.006), hum_db=(-30, -20), drive_db=(3, 7),
    ),
    "EXTREME": dict(
        bandlimit_low=(350, 500), bandlimit_high=(2000, 2800), snr_db=(-5, 5), click_rate=(15.0, 30.0),
        wow_depth=(0.006, 0.012), hum_db=(-24, -14), drive_db=(6, 12),
    ),
}

# canonical composition order, matching brief §7's example
CHAIN_ORDER = ["bandlimit", "nonlinear_distortion", "crackle", "surface_noise", "hum", "wow_flutter"]


def sample_chain_config(tier: str, seed: int, sample_rate: int, include_wow_flutter: bool = False) -> dict:
    """`include_wow_flutter` defaults to False for the main additive-damage benchmark chain.

    Measured finding (see research/limitations.md): even LIGHT-tier wow/flutter depths
    introduce a continuously time-varying misalignment that a single constant-lag alignment
    (soundrevive.metrics.alignment, brief §18) cannot correct, and none of the v1 restoration
    methods attempt time-warp correction either. Leaving it in the default chain made
    SI-SDR/log-spectral-distance dominated by uncorrected timing drift rather than by the
    noise/click/bandlimit damage the benchmark is meant to measure — the metric stopped
    differentiating between methods at all. The brief's own §39 treats wow/flutter correction
    as its own separate experiment (a pitch/time-perturbation model with its own alignment and
    metrics), which is the more correct place for it; set include_wow_flutter=True there.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}, expected one of {TIERS}")
    p = _TIER_PARAMS[tier]
    rng = np.random.default_rng(seed)

    def u(key):
        lo, hi = p[key]
        return float(rng.uniform(lo, hi))

    low_hz = u("bandlimit_low")
    high_hz = min(u("bandlimit_high"), sample_rate / 2 * 0.98)
    steps = [
        {"name": "bandlimit", "params": {"low_hz": low_hz, "high_hz": high_hz, "order": 4}},
        {"name": "nonlinear_distortion", "params": {"drive_db": u("drive_db"), "curve": "tanh"}},
        {"name": "crackle", "params": {"density": u("click_rate"), "amplitude_db": -22.0}},
        {"name": "surface_noise", "params": {"snr_db": u("snr_db"), "color_alpha": 1.0}},
        {"name": "hum", "params": {"freq_hz": 60.0, "n_harmonics": 3, "amp_db": u("hum_db")}},
    ]
    if include_wow_flutter:
        steps.append({"name": "wow_flutter", "params": {"wow_hz": 1.2, "wow_depth": u("wow_depth"), "flutter_hz": 11.0, "flutter_depth": u("wow_depth") / 4}})
    return {"sample_rate": sample_rate, "seed": seed, "tier": tier, "steps": steps}
