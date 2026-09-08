"""Apply a degradation chain config deterministically. The exact resolved config (every
primitive, every param, every derived sub-seed) is what gets persisted alongside the degraded
file (brief §7 'store exact chain'), so nothing is ever silently re-derived differently later.
"""
from __future__ import annotations

import numpy as np

from soundrevive.degradation.primitives import MASK_PRIMITIVES, PRIMITIVES


def _sub_seeds(base_seed: int, n: int) -> list[int]:
    rng = np.random.default_rng(base_seed)
    return [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=n)]


def apply_chain(audio: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (degraded_audio, damage_mask, resolved_config). damage_mask is 1.0 where a
    mask-producing primitive (clicks/crackle/dropout) altered a sample, 0.0 elsewhere — the
    ground truth for declick/dropout-inpainting evaluation."""
    sr = config["sample_rate"]
    steps = config["steps"]
    sub_seeds = _sub_seeds(config["seed"], len(steps))

    out = audio.astype(np.float32).copy()
    mask = np.zeros(len(audio), dtype=np.float32)
    resolved_steps = []
    for step, sub_seed in zip(steps, sub_seeds, strict=True):
        name = step["name"]
        params = dict(step["params"])
        fn = PRIMITIVES[name]
        if name in MASK_PRIMITIVES:
            out, step_mask = fn(out, sr, sub_seed, **params)
            mask = np.maximum(mask, step_mask)
        else:
            out = fn(out, sr, sub_seed, **params)
        resolved_steps.append({"name": name, "params": params, "sub_seed": sub_seed})

    resolved_config = {**config, "steps": resolved_steps}
    return out.astype(np.float32), mask, resolved_config
