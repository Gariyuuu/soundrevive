# SoundRevive — Claims Registry

Rule (brief §54): every headline claim maps to a specific generated artifact. No number in
this file, the paper, or the site may be hand-typed without a corresponding row here pointing
at the script and output file that produced it. Status values: `NOT YET RUN`, `SMOKE ONLY`,
`STANDARD`, `RETRACTED` (with reason, never silently removed).

| ID | Claim | Producing script | Output artifact | Status |
|---|---|---|---|---|
| C1 | Per-method SI-SDR/SDR/SNR-improvement/log-spectral-distance on Regime A | `src/soundrevive/pipeline/run_pipeline.py` (SMOKE), `src/soundrevive/pipeline/run_standard.py` (STANDARD) | `results/benchmark.parquet` (SMOKE, 2 clips/1 speaker), `results/standard/benchmark.parquet` (STANDARD, 24 held-out-speaker clips/6 speakers) | STANDARD — see C6 for the headline finding; SMOKE row's original 2-clip observation still stands for that corpus, see `research/limitations.md` |
| C2 | Degradation determinism (same seed → byte-identical output) | `tests/test_degradation_determinism.py` | pytest pass/fail | STANDARD (test either passes or the claim is false; no partial credit) |
| C3 | Restoration runtime / real-time factor per method | `src/soundrevive/pipeline/run_pipeline.py`, `run_standard.py` | `results/runtime.parquet`, `results/standard/runtime.parquet` | STANDARD — all methods run at real-time-factor << 1 (fastest: passthrough ~1e-6, slowest: stft_mask_denoiser ~0.006) on both corpora |
| C4 | Fidelity vs. quality frontier (RQ2/H2) | not yet written | `results/fidelity.parquet` + a generative method | NOT YET RUN — blocked on a generative-restoration participant (see `research/limitations.md`) |
| C5 | Hallucination proxy panel (RQ4/H4) | `src/soundrevive/metrics/*` (ASR-WER proxy only in v1) | `results/hallucination.parquet`, `results/standard/hallucination.parquet` | STANDARD (ASR-WER leg only) — headline finding: see C6; spectral-energy/ensemble-disagreement proxies still not wired in |
| C6 | Cross-speaker generalization (RQ5) + fidelity-vs-content-preservation divergence (brief §16/§74) | `src/soundrevive/pipeline/run_standard.py` | `results/standard/benchmark.parquet`, `results/standard/hallucination.parquet` | STANDARD — real speaker-held-out test (6 train_pool speakers → 6 never-seen eval_pool speakers, 24 eval clips). Two findings: (1) the discriminative ML denoiser generalizes: it beats every classical baseline on SI-SDR and log-spectral-distance at MODERATE/SEVERE/EXTREME tiers, consistently across held-out speakers (SI-SDR beats Wiener in 12/12 speaker×tier combos at SEVERE+EXTREME). (2) it does NOT generalize on content preservation: it has the *worst* ASR-WER of every method at every tier, and increases WER relative to doing nothing (positive `wer_delta_restoration_introduced`) at every tier — while classical Wiener filtering often *decreases* WER (improves intelligibility) despite a lower SI-SDR than the ML method. Verified per-speaker (not just averages): ML has worse WER-delta than Wiener in 10/12 speaker×tier combos at SEVERE+EXTREME, the same combos where it wins on SI-SDR in 12/12. N=6 held-out speakers/24 clips — real and speaker-general, but still too small for a paper-grade effect-size claim; no bootstrap CI computed yet (§51, not yet implemented). |
| C7 | Ensemble disagreement vs. true error (RQ6/H6) | not yet written | not yet written | NOT YET RUN |
| C8 | Human listening-study preference data (RQ7) | not yet written | `results/listening_study.parquet` | NOT YET RUN — **NO HUMAN DATA EXISTS**; do not report any preference percentage anywhere until this row changes |
| C9 | Historical (Regime B) case-study qualitative notes | not yet written | `research/data_provenance.md` (source cataloging only so far) | NOT YET RUN |
| C10 | Dataset provenance / licensing verification | background research pass (see `docs/HANDOFF.md`) | `research/data_provenance.md` | DONE — 4 clean-corpus candidates + 4 historical-audio sources checked, each with a verified-vs-secondary-source confidence note; open gap (no large rights-clear historical *speech* corpus) documented, not glossed over |
| C11 | Literature matrix (real, verifiable citations only) | background research pass | `research/literature_matrix.csv` | DONE — 25 entries, each with a verification_note; no fabricated citations |

## Hard rule for anyone (human or model) editing the paper or site
If a metric, percentage, ranking, or MOS-style score appears in `paper/` or `site/` and is
not traceable to a row in this table with status `SMOKE ONLY` or `STANDARD`, delete it before
merging. `SMOKE ONLY` numbers must be visually/textually marked as such wherever shown — they
are pipeline-verification artifacts, not benchmark findings, and must never be quoted as if
they were (2 short single-speaker clips cannot support a leaderboard claim).
