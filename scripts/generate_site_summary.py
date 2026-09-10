"""Generates site/app/data/summary.json from the actual results/ artifacts — the site must
not hand-type benchmark numbers (research/claims_registry.md's hard rule applies here too).
Run this whenever results/ changes; it is not run automatically by `next build`.
"""
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    standard_release = json.loads((REPO_ROOT / "results/standard/release.json").read_text())
    smoke_release = json.loads((REPO_ROOT / "results/release.json").read_text())
    bench = pd.read_parquet(REPO_ROOT / "results/standard/benchmark.parquet")
    hallu = pd.read_parquet(REPO_ROOT / "results/standard/hallucination.parquet")

    severe_extreme_bench = bench[bench["tier"].isin(["SEVERE", "EXTREME"])]
    piv_sisdr = severe_extreme_bench.pivot_table(index=["speaker_id", "tier"], columns="method", values="si_sdr_db")
    ml_beats_wiener_sisdr = float((piv_sisdr["stft_mask_denoiser"] > piv_sisdr["wiener"]).mean())

    severe_extreme_hallu = hallu[hallu["tier"].isin(["SEVERE", "EXTREME"])]
    piv_wer = severe_extreme_hallu.pivot_table(index=["speaker_id", "tier"], columns="method", values="wer_delta_restoration_introduced")
    ml_worse_wer_than_wiener = float((piv_wer["stft_mask_denoiser"] > piv_wer["wiener"]).mean())

    tier_summary = (
        bench.groupby(["tier", "method"])[["si_sdr_db", "log_spectral_distance_db"]]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )

    summary = {
        "generated_from": "results/standard/*.parquet + results/release.json (see research/claims_registry.md row C6)",
        "smoke": {
            "n_clips": smoke_release["n_source_clips"],
            "n_conditions": smoke_release["n_conditions"],
        },
        "standard": {
            "n_train_pool_clips": standard_release["n_train_pool_clips"],
            "n_eval_pool_clips": standard_release["n_eval_pool_clips"],
            "n_train_pool_speakers": standard_release["n_train_pool_speakers"],
            "n_eval_pool_speakers": standard_release["n_eval_pool_speakers"],
            "n_tiers": standard_release["n_tiers"],
            "n_methods": standard_release["n_methods"],
            "corpus": standard_release["corpus"],
        },
        "headline_finding": {
            "ml_beats_wiener_on_si_sdr_severe_extreme_fraction": round(ml_beats_wiener_sisdr, 4),
            "ml_worse_wer_delta_than_wiener_severe_extreme_fraction": round(ml_worse_wer_than_wiener, 4),
            "n_speaker_tier_combos_checked": int(len(piv_sisdr)),
        },
        "tier_method_summary": tier_summary,
    }

    out_dir = REPO_ROOT / "site" / "app" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
