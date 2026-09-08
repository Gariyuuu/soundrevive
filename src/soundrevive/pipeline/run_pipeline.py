"""SMOKE-tier end-to-end pipeline: clean -> degrade -> restore (classical + ML) -> align ->
measure -> persist with provenance (brief §79 execution order, §80 completion standard).

This is explicitly a SMOKE-tier run (research/claims_registry.md, research/limitations.md):
2 source clips, 1 reader, 4 degradation tiers. It verifies pipeline *mechanics* and
determinism. It is not sufficient evidence for any comparative-method claim.

Run: `make benchmark` (see Makefile) or `python -m soundrevive.pipeline.run_pipeline`.
"""
from __future__ import annotations

import json
import time
import zlib

import numpy as np
import pandas as pd

from soundrevive.baselines import BASELINES
from soundrevive.degradation import apply_chain, sample_chain_config
from soundrevive.io_utils import load_wav
from soundrevive.metrics import (
    align,
    clipping_fraction,
    log_spectral_distance,
    sdr,
    si_sdr,
    snr_improvement,
    stoi_score,
    transcribe,
    word_error_rate,
)
from soundrevive.ml.denoiser import restore_with_denoiser, save_denoiser, train_denoiser
from soundrevive.pipeline.manifest import REPO_ROOT, clip_path
from soundrevive.provenance import config_hash, git_sha, sha256_of_array


def stable_hash(s: str) -> int:
    """`hash()` on strings is randomized per-process (PYTHONHASHSEED); seeds derived from it
    would break the determinism guarantee tests/test_degradation_determinism.py checks for."""
    return zlib.crc32(s.encode("utf-8"))


TIERS = ["LIGHT", "MODERATE", "SEVERE", "EXTREME"]
SEGMENT_S = 4.0
TRAIN_CLIP = "alice00_16k_clip"
HELDOUT_CLIP = "alice01_16k_clip"

METHOD_META = {
    "passthrough": {"category": "SIGNAL PROCESSING", "content_label": "UNPROCESSED"},
    "spectral_gate": {"category": "SIGNAL PROCESSING", "content_label": "RESTORED"},
    "wiener": {"category": "SIGNAL PROCESSING", "content_label": "RESTORED"},
    "median_declick": {"category": "SIGNAL PROCESSING", "content_label": "RECONSTRUCTED"},
    "stft_mask_denoiser": {"category": "DISCRIMINATIVE ML", "content_label": "RESTORED"},
}


def _segment(audio: np.ndarray, sr: int, seg_idx: int, seg_len_s: float = SEGMENT_S) -> np.ndarray:
    n = int(seg_len_s * sr)
    start = seg_idx * n
    return audio[start:start + n]


def _degrade(audio: np.ndarray, sr: int, tier: str, seed: int):
    cfg = sample_chain_config(tier, seed=seed, sample_rate=sr)
    degraded, mask, resolved_cfg = apply_chain(audio, cfg)
    return degraded, mask, resolved_cfg


def build_training_pairs(sr: int) -> list[tuple[np.ndarray, np.ndarray]]:
    clean_full, sr_loaded = load_wav(str(clip_path(TRAIN_CLIP)))
    assert sr_loaded == sr
    pairs = []
    for seg_idx in range(4):  # reserve segment 4 (16-20s) as a held-out segment for eval
        clean_seg = _segment(clean_full, sr, seg_idx)
        for tier in TIERS:
            seed = 1000 * seg_idx + stable_hash(tier) % 997 + 1
            degraded_seg, _, _ = _degrade(clean_seg, sr, tier, seed)
            pairs.append((clean_seg, degraded_seg))
    return pairs


def run_smoke_pipeline() -> dict:
    sr = 16000
    t0 = time.time()

    print("[1/5] building training pairs from augmented degradations of the train clip...")
    train_pairs = build_training_pairs(sr)
    print(f"      {len(train_pairs)} pairs")

    print("[2/5] training STFT-mask discriminative denoiser (SMOKE scope, CPU)...")
    model = train_denoiser(train_pairs, epochs=25, seed=0)
    weights_dir = REPO_ROOT / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "stft_mask_denoiser_smoke.pt"
    save_denoiser(model, str(weights_path))
    model_hash = sha256_of_array(np.frombuffer(weights_path.read_bytes(), dtype=np.uint8))

    eval_conditions = []
    clean_train_full, _ = load_wav(str(clip_path(TRAIN_CLIP)))
    heldout_segment_clean = _segment(clean_train_full, sr, 4)
    clean_heldout_full, _ = load_wav(str(clip_path(HELDOUT_CLIP)))
    eval_conditions.append(("train_recording_heldout_segment", TRAIN_CLIP, heldout_segment_clean))
    eval_conditions.append(("heldout_recording_full", HELDOUT_CLIP, clean_heldout_full))

    print("[3/5] transcribing clean references (ASR baseline for hallucination proxy)...")
    clean_transcripts = {name: transcribe(clean, sr) for name, _, clean in eval_conditions}
    for name, txt in clean_transcripts.items():
        print(f"      {name}: {txt[:80]!r}")

    print("[4/5] degrading + restoring + measuring across tiers and methods...")
    benchmark_rows = []
    hallucination_rows = []
    runtime_rows = []
    degradation_rows = []

    eval_seed_base = 999_001
    for cond_name, clip_id, clean in eval_conditions:
        for ti, tier in enumerate(TIERS):
            seed = eval_seed_base + 17 * ti + stable_hash(cond_name) % 991
            degraded, damage_mask, resolved_cfg = _degrade(clean, sr, tier, seed)

            degradation_rows.append({
                "condition": cond_name, "clip_id": clip_id, "tier": tier, "seed": seed,
                "config_hash": config_hash(resolved_cfg), "damage_mask_fraction": float(damage_mask.mean()),
                "resolved_config_json": json.dumps(resolved_cfg),
            })

            t = time.time()
            method_outputs = {"passthrough": BASELINES["passthrough"](degraded, sr)}
            runtime_rows.append(_runtime_row(cond_name, tier, "passthrough", clean, time.time() - t))

            t = time.time()
            method_outputs["spectral_gate"] = BASELINES["spectral_gate"](degraded, sr)
            runtime_rows.append(_runtime_row(cond_name, tier, "spectral_gate", clean, time.time() - t))

            t = time.time()
            method_outputs["wiener"] = BASELINES["wiener"](degraded, sr)
            runtime_rows.append(_runtime_row(cond_name, tier, "wiener", clean, time.time() - t))

            t = time.time()
            method_outputs["median_declick"] = BASELINES["median_declick"](degraded, sr)
            runtime_rows.append(_runtime_row(cond_name, tier, "median_declick", clean, time.time() - t))

            t = time.time()
            method_outputs["stft_mask_denoiser"] = restore_with_denoiser(model, degraded)
            runtime_rows.append(_runtime_row(cond_name, tier, "stft_mask_denoiser", clean, time.time() - t))

            degraded_txt = transcribe(degraded, sr)
            degraded_wer = word_error_rate(degraded_txt, clean_transcripts[cond_name])

            for method_name, restored in method_outputs.items():
                aligned_est, aligned_ref, lag = align(restored, clean)
                aligned_deg, aligned_ref_d, _ = align(degraded, clean)

                row = {
                    "condition": cond_name, "clip_id": clip_id, "tier": tier, "method": method_name,
                    "method_category": METHOD_META[method_name]["category"],
                    "content_label": METHOD_META[method_name]["content_label"],
                    "si_sdr_db": si_sdr(aligned_est, aligned_ref),
                    "sdr_db": sdr(aligned_est, aligned_ref),
                    "si_sdr_degraded_db": si_sdr(aligned_deg, aligned_ref_d),
                    "snr_improvement_db": snr_improvement(aligned_deg, aligned_est, aligned_ref),
                    "log_spectral_distance_db": log_spectral_distance(aligned_est, aligned_ref),
                    "stoi": stoi_score(aligned_est, aligned_ref, sr),
                    "clipping_fraction": clipping_fraction(restored),
                    "alignment_lag_samples": lag,
                    "seed": seed, "config_hash": config_hash(resolved_cfg),
                }
                benchmark_rows.append(row)

                restored_txt = transcribe(restored, sr)
                restored_wer = word_error_rate(restored_txt, clean_transcripts[cond_name])
                hallucination_rows.append({
                    "condition": cond_name, "clip_id": clip_id, "tier": tier, "method": method_name,
                    "method_category": METHOD_META[method_name]["category"],
                    "asr_transcript": restored_txt,
                    "wer_vs_clean_reference": restored_wer,
                    "wer_of_degraded_input_vs_clean_reference": degraded_wer,
                    "wer_delta_restoration_introduced": (
                        restored_wer - degraded_wer if not (np.isnan(restored_wer) or np.isnan(degraded_wer)) else float("nan")
                    ),
                })

    print("[5/5] writing result artifacts...")
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    benchmark_df = pd.DataFrame(benchmark_rows)
    hallucination_df = pd.DataFrame(hallucination_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    degradation_df = pd.DataFrame(degradation_rows)

    benchmark_df.to_parquet(results_dir / "benchmark.parquet", index=False)
    hallucination_df.to_parquet(results_dir / "hallucination.parquet", index=False)
    runtime_df.to_parquet(results_dir / "runtime.parquet", index=False)
    degradation_df.to_parquet(results_dir / "degradation_breakdown.parquet", index=False)
    fidelity_cols = ["condition", "clip_id", "tier", "method", "method_category", "si_sdr_db", "sdr_db",
                      "log_spectral_distance_db", "stoi"]
    benchmark_df[fidelity_cols].to_parquet(results_dir / "fidelity.parquet", index=False)

    release = {
        "tier": "SMOKE",
        "git_sha": git_sha(),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_source_clips": 2,
        "n_conditions": len(eval_conditions),
        "n_tiers": len(TIERS),
        "n_methods": len(METHOD_META),
        "model_weights_sha256": model_hash,
        "training_pairs": len(train_pairs),
        "training_epochs": 25,
        "total_runtime_s": time.time() - t0,
        "clean_transcripts": clean_transcripts,
        "warning": (
            "SMOKE-tier release. 2 source clips, 1 reader. Do not treat as a STANDARD-tier "
            "benchmark finding — see research/claims_registry.md and research/limitations.md."
        ),
    }
    with open(results_dir / "release.json", "w") as f:
        json.dump(release, f, indent=2)

    print(f"done in {release['total_runtime_s']:.1f}s")
    return release


def _runtime_row(condition, tier, method, clean_ref, elapsed_s):
    duration_s = len(clean_ref) / 16000.0
    return {
        "condition": condition, "tier": tier, "method": method,
        "audio_duration_s": duration_s, "wall_time_s": elapsed_s,
        "real_time_factor": elapsed_s / duration_s if duration_s > 0 else float("nan"),
    }


if __name__ == "__main__":
    run_smoke_pipeline()
