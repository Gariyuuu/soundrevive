"""STANDARD-tier pipeline: same clean -> degrade -> restore -> align -> measure -> persist
mechanics as run_pipeline.py (SMOKE), but on the LibriSpeech dev-clean subset
(data/manifest.json, 12 speakers, verified CC BY 4.0) with a real train/eval SPEAKER split —
the discriminative ML denoiser only ever sees train_pool speakers during training, and is
evaluated on eval_pool speakers it has never heard, which is a genuine speaker-generalization
test (RQ5), unlike SMOKE's same-speaker held-out-segment/held-out-recording conditions.

Ground-truth transcripts are the real human LibriSpeech transcripts (data/manifest.json
`transcript_reference`), not ASR pseudo-labels — a stronger content-preservation reference
than SMOKE's whisper-on-clean-audio approach, at the cost of only being available for this
corpus.

Run: `make benchmark-standard` (see Makefile) or `python -m soundrevive.pipeline.run_standard`.
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
from soundrevive.pipeline.manifest import REPO_ROOT, load_manifest
from soundrevive.provenance import config_hash, git_sha, sha256_of_array

TIERS = ["LIGHT", "MODERATE", "SEVERE", "EXTREME"]
TRAIN_EPOCHS = 15  # lower than SMOKE's 25: ~4x more training audio per epoch at STANDARD scale

METHOD_META = {
    "passthrough": {"category": "SIGNAL PROCESSING", "content_label": "UNPROCESSED"},
    "spectral_gate": {"category": "SIGNAL PROCESSING", "content_label": "RESTORED"},
    "wiener": {"category": "SIGNAL PROCESSING", "content_label": "RESTORED"},
    "median_declick": {"category": "SIGNAL PROCESSING", "content_label": "RECONSTRUCTED"},
    "stft_mask_denoiser": {"category": "DISCRIMINATIVE ML", "content_label": "RESTORED"},
}


def stable_hash(s: str) -> int:
    """`hash()` on strings is randomized per-process (PYTHONHASHSEED); a stable hash keeps
    seeds (and therefore degradation configs) reproducible across runs/machines."""
    return zlib.crc32(s.encode("utf-8"))


def _librispeech_entries(manifest: dict) -> tuple[list[dict], list[dict]]:
    ls_entries = [e for e in manifest["clean_speech"] if e["clip_id"].startswith("ls_")]
    train = [e for e in ls_entries if e["speaker_split"] == "train_pool"]
    ev = [e for e in ls_entries if e["speaker_split"] == "eval_pool"]
    return train, ev


def _degrade(audio: np.ndarray, sr: int, tier: str, seed: int):
    cfg = sample_chain_config(tier, seed=seed, sample_rate=sr)
    degraded, mask, resolved_cfg = apply_chain(audio, cfg)
    return degraded, mask, resolved_cfg


def build_training_pairs(train_entries: list[dict], sr: int) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for entry in train_entries:
        clean, sr_loaded = load_wav(str(REPO_ROOT / entry["path"]))
        assert sr_loaded == sr
        for tier in TIERS:
            seed = stable_hash(f"{entry['clip_id']}::{tier}::train") % 1_000_000
            degraded, _, _ = _degrade(clean, sr, tier, seed)
            pairs.append((clean, degraded))
    return pairs


def _runtime_row(condition, tier, method, clean_ref, sr, elapsed_s):
    duration_s = len(clean_ref) / sr
    return {
        "condition": condition, "tier": tier, "method": method,
        "audio_duration_s": duration_s, "wall_time_s": elapsed_s,
        "real_time_factor": elapsed_s / duration_s if duration_s > 0 else float("nan"),
    }


def run_standard_pipeline() -> dict:
    sr = 16000
    t0 = time.time()
    manifest = load_manifest()
    train_entries, eval_entries = _librispeech_entries(manifest)
    print(f"[0/5] {len(train_entries)} train_pool clips ({sorted({e['speaker_id'] for e in train_entries}, key=int)}), "
          f"{len(eval_entries)} eval_pool clips ({sorted({e['speaker_id'] for e in eval_entries}, key=int)})")

    print("[1/5] building training pairs from augmented degradations of train_pool clips...")
    train_pairs = build_training_pairs(train_entries, sr)
    print(f"      {len(train_pairs)} pairs, {sum(len(c) for c, _ in train_pairs) / sr:.1f}s of clean audio x {len(TIERS)} tiers")

    print(f"[2/5] training STFT-mask discriminative denoiser (STANDARD scope, CPU, {TRAIN_EPOCHS} epochs)...")
    t_train = time.time()
    model = train_denoiser(train_pairs, epochs=TRAIN_EPOCHS, seed=0)
    print(f"      trained in {time.time() - t_train:.1f}s")
    weights_dir = REPO_ROOT / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "stft_mask_denoiser_standard.pt"
    save_denoiser(model, str(weights_path))
    model_hash = sha256_of_array(np.frombuffer(weights_path.read_bytes(), dtype=np.uint8))

    print(f"[3/5] degrading + restoring + measuring {len(eval_entries)} eval_pool clips x {len(TIERS)} tiers "
          f"x {len(METHOD_META)} methods (speaker-held-out generalization test)...")
    benchmark_rows, hallucination_rows, runtime_rows, degradation_rows = [], [], [], []

    for ei, entry in enumerate(eval_entries):
        clean, sr_loaded = load_wav(str(REPO_ROOT / entry["path"]))
        assert sr_loaded == sr
        clip_id = entry["clip_id"]
        speaker_id = entry["speaker_id"]
        reference_text = entry["transcript_reference"]

        for tier in TIERS:
            seed = stable_hash(f"{clip_id}::{tier}::eval") % 1_000_000
            degraded, damage_mask, resolved_cfg = _degrade(clean, sr, tier, seed)

            degradation_rows.append({
                "condition": clip_id, "clip_id": clip_id, "speaker_id": speaker_id, "tier": tier, "seed": seed,
                "config_hash": config_hash(resolved_cfg), "damage_mask_fraction": float(damage_mask.mean()),
                "resolved_config_json": json.dumps(resolved_cfg),
            })

            method_outputs = {}
            for method_name in ["passthrough", "spectral_gate", "wiener", "median_declick"]:
                t = time.time()
                method_outputs[method_name] = BASELINES[method_name](degraded, sr)
                runtime_rows.append(_runtime_row(clip_id, tier, method_name, clean, sr, time.time() - t))
            t = time.time()
            method_outputs["stft_mask_denoiser"] = restore_with_denoiser(model, degraded)
            runtime_rows.append(_runtime_row(clip_id, tier, "stft_mask_denoiser", clean, sr, time.time() - t))

            degraded_txt = transcribe(degraded, sr)
            degraded_wer = word_error_rate(degraded_txt, reference_text)

            for method_name, restored in method_outputs.items():
                aligned_est, aligned_ref, lag = align(restored, clean)
                aligned_deg, aligned_ref_d, _ = align(degraded, clean)

                benchmark_rows.append({
                    "condition": clip_id, "clip_id": clip_id, "speaker_id": speaker_id, "tier": tier,
                    "method": method_name, "method_category": METHOD_META[method_name]["category"],
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
                })

                restored_txt = transcribe(restored, sr)
                restored_wer = word_error_rate(restored_txt, reference_text)
                hallucination_rows.append({
                    "condition": clip_id, "clip_id": clip_id, "speaker_id": speaker_id, "tier": tier,
                    "method": method_name, "method_category": METHOD_META[method_name]["category"],
                    "asr_transcript": restored_txt, "reference_transcript": reference_text,
                    "wer_vs_clean_reference": restored_wer,
                    "wer_of_degraded_input_vs_clean_reference": degraded_wer,
                    "wer_delta_restoration_introduced": (
                        restored_wer - degraded_wer if not (np.isnan(restored_wer) or np.isnan(degraded_wer)) else float("nan")
                    ),
                })
        print(f"      [{ei + 1}/{len(eval_entries)}] {clip_id} (speaker {speaker_id}) done, {time.time() - t0:.0f}s elapsed")

    print("[4/5] writing result artifacts...")
    results_dir = REPO_ROOT / "results" / "standard"
    results_dir.mkdir(parents=True, exist_ok=True)

    benchmark_df = pd.DataFrame(benchmark_rows)
    hallucination_df = pd.DataFrame(hallucination_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    degradation_df = pd.DataFrame(degradation_rows)

    benchmark_df.to_parquet(results_dir / "benchmark.parquet", index=False)
    hallucination_df.to_parquet(results_dir / "hallucination.parquet", index=False)
    runtime_df.to_parquet(results_dir / "runtime.parquet", index=False)
    degradation_df.to_parquet(results_dir / "degradation_breakdown.parquet", index=False)
    fidelity_cols = ["condition", "clip_id", "speaker_id", "tier", "method", "method_category",
                      "si_sdr_db", "sdr_db", "log_spectral_distance_db", "stoi"]
    benchmark_df[fidelity_cols].to_parquet(results_dir / "fidelity.parquet", index=False)

    release = {
        "tier": "STANDARD",
        "git_sha": git_sha(),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus": "LibriSpeech dev-clean subset (OpenSLR 12, CC BY 4.0)",
        "n_train_pool_clips": len(train_entries),
        "n_eval_pool_clips": len(eval_entries),
        "n_train_pool_speakers": len({e["speaker_id"] for e in train_entries}),
        "n_eval_pool_speakers": len({e["speaker_id"] for e in eval_entries}),
        "speaker_generalization": "eval_pool speakers never appear in any ML-denoiser training pair",
        "n_tiers": len(TIERS),
        "n_methods": len(METHOD_META),
        "model_weights_sha256": model_hash,
        "training_pairs": len(train_pairs),
        "training_epochs": TRAIN_EPOCHS,
        "total_runtime_s": time.time() - t0,
        "warning": (
            "STANDARD-tier release: 48 clips, 12 speakers, real speaker-held-out "
            "generalization split. Still a modest N for bootstrap CIs across clips/speakers "
            "-- see research/limitations.md before treating any single-clip difference as "
            "significant. Ground-truth transcripts are real human LibriSpeech labels, not "
            "ASR pseudo-labels."
        ),
    }
    with open(results_dir / "release.json", "w") as f:
        json.dump(release, f, indent=2)

    print(f"done in {release['total_runtime_s']:.1f}s")
    return release


if __name__ == "__main__":
    run_standard_pipeline()
