# SoundRevive — Claims Registry

Rule (brief §54): every headline claim maps to a specific generated artifact. No number in
this file, the paper, or the site may be hand-typed without a corresponding row here pointing
at the script and output file that produced it. Status values: `NOT YET RUN`, `SMOKE ONLY`,
`STANDARD`, `RETRACTED` (with reason, never silently removed).

| ID | Claim | Producing script | Output artifact | Status |
|---|---|---|---|---|
| C1 | Per-method SI-SDR/SDR/SNR-improvement/log-spectral-distance on Regime A, smoke set | `src/soundrevive/pipeline/run_pipeline.py` | `results/benchmark.parquet` | SMOKE ONLY — 2 source clips, 1 speaker; contains a real SI-SDR/log-spectral-distance disagreement for the ML method, see `research/limitations.md` "What the first SMOKE-tier run actually found" |
| C2 | Degradation determinism (same seed → byte-identical output) | `tests/test_degradation_determinism.py` | pytest pass/fail | STANDARD (test either passes or the claim is false; no partial credit) |
| C3 | Restoration runtime / real-time factor per method | `src/soundrevive/pipeline/run_pipeline.py` | `results/runtime.parquet` | SMOKE ONLY |
| C4 | Fidelity vs. quality frontier (RQ2/H2) | not yet written | `results/fidelity.parquet` + a generative method | NOT YET RUN — blocked on a generative-restoration participant (see `research/limitations.md`) |
| C5 | Hallucination proxy panel (RQ4/H4) | `src/soundrevive/metrics/*` (fidelity/spectral proxy only in v1) | `results/hallucination.parquet` | SMOKE ONLY — ASR-divergence proxy only; spectral/ensemble proxies not yet wired into the pipeline runner |
| C6 | Generalization across unseen degradation (RQ5) | not yet written | not yet written | NOT YET RUN |
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
