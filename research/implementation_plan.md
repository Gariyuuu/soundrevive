# SoundRevive — Implementation Plan

Execution order follows brief §79: audit → literature → data licensing → clean benchmark
selection → degradation engine → classical baselines → first ML method → objective
evaluation → ASR/content preservation → additional models → hallucination analysis →
generalization → real historical cases → figures → web explorer → paper → tests →
Reviewer 2 → release → deploy.

## Phase 0 — Audit (done)
No pre-existing repo; fresh build. Environment: macOS, Python 3.11 venv (project-local,
not global — global `python3` is 3.14 with no packages, and a stray dependency on system
Python has bitten other projects in this workspace before), Node v26, ffmpeg present, ~12–14
GB free disk at start (constrains model-weight choices — see `research/limitations.md`).

## Phase 1 — Repo scaffold + research docs (done, this pass)
`research/*.md`, `docs/HANDOFF.md`, directory skeleton, git init.

## Phase 2 — Data licensing + clean reference (done for SMOKE tier, in progress for STANDARD)
SMOKE: 2 CC0 LibriVox clips fetched and hash-recorded (`data/raw/clean_speech/`,
`data/manifest.json`). STANDARD: candidate corpora and historical sources under review in a
background research pass; see `research/data_provenance.md` and `docs/HANDOFF.md` for status
at hand-off time.

## Phase 3 — Degradation engine (this pass)
`src/soundrevive/degradation/`: primitives, tiers, deterministic chains, config
serialization. Golden-fixture determinism tests.

## Phase 4 — Classical baselines (this pass)
`src/soundrevive/baselines/`: passthrough, bandpass, spectral gating, Wiener, median declick.

## Phase 5 — First ML method (this pass, SMOKE scope)
`src/soundrevive/ml/`: STFT-mask discriminative denoiser, trained on augmented pairs from the
2-clip corpus. Explicitly labeled SMOKE-tier, not a generalization claim.

## Phase 6 — Objective evaluation (this pass)
`src/soundrevive/metrics/`: SI-SDR/SDR/SNR-improvement/log-spectral distance, STOI, alignment,
clipping check. Wired into `src/soundrevive/pipeline/run_pipeline.py`.

## Phase 7 — ASR/content preservation (this pass, minimal)
Whisper-tiny WER between clean/degraded/restored transcripts, wired as one of the
hallucination proxies.

## Phase 8+ — Not yet done this pass (tracked in `docs/HANDOFF.md`)
Additional ML methods (esp. a generative/strength-controllable one — required for the
flagship fidelity-vs-quality claim), full hallucination panel wiring, generalization splits,
Regime B historical case studies, figures, the Next.js explorer site, the paper, CI, and
Reviewer-2 pass. `docs/HANDOFF.md` is the canonical live status; this plan is not re-edited
per session — HANDOFF is.

## Compute tiers (brief §59)
- **SMOKE** (implemented this pass): 2 clips, ~40s audio, CPU, single-digit-minute runtime.
  Purpose: verify pipeline mechanics and determinism only.
- **STANDARD**: full clean corpus (pending Phase 2 completion), all baselines + all ML
  methods, full metric panel, generalization splits. Not yet run.
- **EXTENDED**: additional models/degradations beyond STANDARD. Not yet scoped.
