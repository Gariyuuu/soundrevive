# SoundRevive — HANDOFF

Canonical, living status document. Read this first in any new session — it is more current
than `research/implementation_plan.md`, which records the plan, not day-to-day state.

Last updated: 2026-09-08 (repo's first build session).

## One-paragraph status

The repo now has a real, tested, deterministic, end-to-end SMOKE-tier pipeline: 2 CC0
LibriVox clips → historically-motivated degradation chains (4 tiers) → 4 classical baselines
+ 1 discriminative ML denoiser → alignment → SI-SDR/SDR/log-spectral-distance/STOI/ASR-WER
metrics → provenance-tagged Parquet/JSON results. 22 tests pass, lint is clean. A separate
research pass verified real dataset licensing (`research/data_provenance.md`) and compiled a
25-entry real literature matrix (`research/literature_matrix.csv`). Everything from brief §41
onward (the Next.js site), the paper, figures, a generative-restoration method, the
hallucination-proxy panel beyond ASR-WER, generalization splits, Regime B case studies, CI's
frontend leg, and deployment are **not built yet**. This was one build session's worth of
work on an 80-section brief; treat it as Phase 0-7 of `research/implementation_plan.md` done,
Phase 8+ open.

## Environment notes (read before running anything)

- **Python**: use the project-local venv at `.venv` (created with `python3.11`, not the
  system `python3`, which resolves to 3.14 with no packages installed — see
  `research/implementation_plan.md` Phase 0). `source .venv/bin/activate` before any
  `python`/`pytest`/`make` command, or use `make setup` to (re)create it.
- **Disk**: the dev machine had ~12-14 GB free at session start. torch (CPU wheel) +
  torchaudio + whisper-tiny.en (~72 MB, auto-downloaded to `~/.cache/whisper` on first
  `transcribe()` call) + numpy/scipy/pandas/pyarrow fit comfortably, but **do not add a large
  pretrained checkpoint (e.g. a diffusion restoration model) without checking free disk
  first** and recording its size/hash in `research/limitations.md`.
- **`hash()` on strings is randomized per process** (`PYTHONHASHSEED`) — never use it to
  derive a seed. `soundrevive.pipeline.run_pipeline.stable_hash` (zlib.crc32-based) is the
  pattern to reuse; this was a real bug caught in this session (see below).

## What's actually built (verified, not aspirational)

| Area | File(s) | Status |
|---|---|---|
| Degradation primitives | `src/soundrevive/degradation/primitives.py` | 10 primitives, deterministic, tested |
| Tiers + canonical chain | `src/soundrevive/degradation/tiers.py` | 4 tiers; wow/flutter excluded from the default chain (see "Findings" below) |
| Classical baselines | `src/soundrevive/baselines/` | passthrough, spectral_gate, wiener, median_declick — all tested against real speech, not just synthetic tones |
| Discriminative ML | `src/soundrevive/ml/denoiser.py` | STFT-mask conv model, trained on augmented pairs from the 2-clip corpus |
| Metrics | `src/soundrevive/metrics/` | SI-SDR, SDR, SNR-improvement, log-spectral-distance, STOI, alignment (FFT cross-correlation), clipping, ASR-WER (whisper-tiny.en) |
| Pipeline | `src/soundrevive/pipeline/run_pipeline.py` | full SMOKE run: 2 clips × 4 tiers × 5 methods, writes `results/*.parquet` + `release.json` |
| Data | `data/manifest.json`, `data/raw/clean_speech/*.wav` | 2 CC0 LibriVox clips, provenance recorded, hash-checked by a test |
| Tests | `tests/` | 22 tests: determinism, metrics correctness, baseline effectiveness on real speech, manifest/provenance integrity |
| Lint/CI | `pyproject.toml` (ruff config), `.github/workflows/ci.yml` | ruff + pytest only; no frontend job yet |
| Research docs | `research/*.md`, `research/literature_matrix.csv` | see below |

Run it: `make setup && make test && make benchmark` (takes ~1-2 min on CPU; first run also
downloads whisper-tiny.en, ~72 MB).

## Bugs found and fixed in this session (worth knowing before touching this code)

1. **STFT/ISTFT energy amplification bug** (`src/soundrevive/stft_utils.py`): a hand-rolled
   overlap-add reconstruction was only an exact inverse for an *unmodified* spectrum.
   Applying it to a magnitude-masked spectrum (what spectral gating/Wiener/the ML denoiser
   all produce) amplified reconstructed RMS 3-14x in testing. Fixed by wrapping
   `scipy.signal.stft/istft` instead of re-deriving the normalization by hand. If you ever
   reconstruct audio from a modified spectrum anywhere in this codebase, use
   `soundrevive.stft_utils.stft/istft`, not a new hand-rolled version.
2. **Bandlimit tier-parameter bug** (`src/soundrevive/degradation/tiers.py`): the LIGHT
   tier's low-cutoff sampling range could exceed the EXTREME tier's, occasionally making
   "LIGHT" degradation filter out just as much content as "EXTREME." Fixed by sampling
   `bandlimit_low` and `bandlimit_high` from separate, tier-appropriate ranges instead of
   deriving the high edge from the low edge.
3. **Median declick threshold too aggressive on real speech**
   (`src/soundrevive/baselines/declick.py`): a single global MAD threshold flagged ~17% of
   ordinary speech samples as "clicks" (plosives/sibilants trigger it), making output worse
   than the input. Fixed with a locally-adaptive threshold (a second, wider median filter
   over the residual magnitude, so quiet vs. loud passages get different noise floors).
4. **`hash()`-based seed derivation** in an early draft of `run_pipeline.py` would have
   silently broken determinism across Python process runs (see Environment notes above).
   Replaced with `zlib.crc32`-based `stable_hash`.
5. **Wow/flutter swamped fidelity metrics.** It was in the default canonical chain per the
   brief's §7 example. Even LIGHT-tier depths introduce continuous time-varying misalignment
   that the alignment step (a single constant lag) can't correct, and no v1 restoration
   method attempts time-warp correction — SI-SDR stopped differentiating between *any*
   methods (everything scored deeply negative and roughly equal). Removed from the default
   chain; kept available via `include_wow_flutter=True` for a dedicated experiment, which is
   where the brief's own §39 already says wow/flutter correction belongs.

All five are documented in more depth in `research/limitations.md` under "What the first
SMOKE-tier run actually found" — read that section before trusting or extending the current
numbers.

## A real (if tiny-sample) finding from the corrected SMOKE run

The discriminative ML denoiser and log-spectral-distance disagree with SI-SDR about whether
it helped: SI-SDR sometimes improves at SEVERE/EXTREME tiers (up to +6-7 dB improvement vs.
passthrough) while log-spectral-distance is consistently ~2-3x worse than every classical
baseline at every tier, and at LIGHT/MODERATE tiers the ML method actively *hurts* SI-SDR
relative to doing nothing. This is directly relevant to RQ2/H2 (fidelity vs. quality
divergence) and brief §74 (metric disagreement) — but it's 2 clips/1 reader, so it is
reported as an observation, not a finding; see `research/claims_registry.md` row C1.

## What's NOT built (the majority of the brief)

Roughly in the order `research/implementation_plan.md` recommends tackling them:

1. **A generative-restoration method.** Nothing in v1 has a strength/aggressiveness
   parameter. This blocks RQ2/RQ3/RQ4/H2/H3/H4 and the paper's own flagship framing
   ("cleaner is not closer") — without a method that can trade fidelity for perceived
   quality, there's no fidelity-vs-quality frontier to plot. **This is the single highest-
   leverage next step.**
2. **STANDARD-tier clean corpus.** `research/data_provenance.md` recommends LJSpeech (true
   public domain, verified) as primary, LibriSpeech (CC BY 4.0, verified) as secondary. Needs
   a download script + manifest entries + re-running the pipeline at STANDARD scale.
3. **Full hallucination-proxy panel.** Only the ASR-WER leg is wired into `run_pipeline.py`.
   The spectral-energy-outside-reference and ensemble-disagreement proxies described in
   `research/restoration_taxonomy.md` §4 need their own metric functions and wiring.
4. **Generalization splits (RQ5)** — train/tune on one degradation distribution, eval on
   unseen noise/bandwidth/severity/recording class. Not attempted; the SMOKE run's "held-out
   recording" condition is the closest thing so far, and it's same-speaker/same-session.
5. **Regime B (genuine historical recordings).** `research/data_provenance.md` identifies
   UCSB Cylinder Audio Archive (pre-1923 items, explicitly free for any use) as the safest
   redistributable historical source, and flags that no large rights-clear corpus of genuine
   historical *speech* exists — Regime B speech will be small-volume. Nothing downloaded yet.
6. **Figures, paper, `/paper` and `/research-methods` site routes** (brief §46-49).
7. **The Next.js site** (brief §41-45) — `/`, `/benchmark`, `/listen`, `/degradations`,
   `/methods`, `/fidelity`, `/hallucination`, `/historical`, `/study`. Nothing scaffolded.
   `site/` directory exists but is empty.
8. **Human listening study.** Infrastructure not built. Per brief §32/§54/`research/
   claims_registry.md` row C8: **no human data exists**; do not report any preference number
   until this changes, and label any future pipeline test explicitly `SIMULATED PIPELINE`.
9. **CI frontend leg, Vercel deployment.** Nothing to deploy yet (no site).
10. **Reviewer-2 pass** (brief §77) — premature before the above exists.

## Suggested next session's first move

Either (a) add a generative-restoration participant (even a small strength-controllable
model — e.g. a diffusion or VAE-based bandwidth-extension/inpainting model with a tunable
guidance/strength parameter) so RQ2/RQ3/H2/H3 become answerable, or (b) scale to the
STANDARD-tier LJSpeech corpus so the existing classical+discriminative comparison has more
than 2 clips behind it. (a) unlocks more of the brief's actual scientific point; (b) is lower
risk and makes every existing number more trustworthy. Recommend (b) first since it's
lower-risk and makes (a)'s eventual results more trustworthy too — but this is a judgment
call, not dictated by the brief.
