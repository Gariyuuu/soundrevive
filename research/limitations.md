# SoundRevive — Limitations (living document)

Updated as of the first pipeline run. This file is a commitment device: it must be read and
updated before any claim is promoted from `SMOKE ONLY` to `STANDARD` in
`research/claims_registry.md`.

## Data
- **SMOKE corpus**: 2 clips (~40s total), single speaker, single recording session
  (public-domain LibriVox recordings of "New Adventures of Alice" read by John Rae, CC0,
  verified via the Internet Archive metadata API — see `research/data_provenance.md`). This
  is a pipeline-verification corpus only; it cannot support any generalization,
  ranking-stability, or fidelity-vs-quality claim.
- **STANDARD corpus**: 48 clips (~6.1 min total), 12 speakers, from LibriSpeech dev-clean
  (OpenSLR 12, CC BY 4.0, verified — `research/data_provenance.md` §1.2). Split into
  `train_pool` (6 speakers, 24 clips — used only to build ML-denoiser training pairs) and
  `eval_pool` (6 speakers, 24 clips — held out entirely, never seen during training), a real
  speaker-generalization split (`data/manifest.json`'s `librispeech_speaker_split`). This
  supports the RQ5 cross-speaker generalization result in `research/claims_registry.md` row
  C6, but 6 held-out speakers / 24 clips is still a modest N — enough to see a consistent
  per-speaker pattern (checked directly, not just averaged), not enough for a paper-grade
  bootstrap effect size (§51 statistics not yet implemented).
  `research/data_provenance.md` also recommends **LJSpeech** (verified true public domain) as
  a further STANDARD-tier candidate — not yet fetched, since LibriSpeech's native
  multi-speaker split was the more direct fit for RQ5. It found no large, unambiguously
  rights-clear corpus of genuine early-20th-century *speech* (as opposed to music) for Regime
  B — see that file's "Open gap" note; Regime B speech will likely be small-volume UCSB
  pre-1923 cylinder material plus Library-of-Congress items usable only for
  reference/no-redistribution evaluation (LOC's terms are more restrictive than their
  post-2022 public-domain status alone would suggest).
- LibriVox recordings, while legally clean (public domain / CC0), are **not acoustically
  pristine studio recordings** — they may contain room tone, breath noise, or amateur mic
  chains. They are adequate as a "clean-ish reference" for engineering validation but a
  STANDARD-tier fidelity benchmark should prefer a studio-quality corpus (e.g. LJSpeech,
  LibriSpeech test-clean) once licensing/redistribution terms for demo clips are confirmed.
- Regime B (genuine historical recordings) is not yet populated with case studies; only
  candidate sources have been identified (`research/data_provenance.md`).

## Methods
- v1 ships classical baselines (spectral gating, Wiener, bandpass, median declick) and one
  discriminative ML method (STFT-mask conv denoiser). **No generative-restoration method
  exists yet.** This blocks full evaluation of RQ2/RQ4/H2/H4 — the project's flagship
  "cleaner is not closer" framing needs a method that can trade fidelity for perceived
  quality via a strength parameter, and none is implemented yet.
- The discriminative model's cross-speaker generalization (RQ5) HAS now been tested (STANDARD
  run, 6 train speakers → 6 disjoint held-out eval speakers) — see
  `research/claims_registry.md` row C6. It has not been tested against unseen recording
  conditions (e.g. a different corpus entirely) or unseen degradation *types* (only unseen
  degradation *parameter draws* from the same tier distributions it trained on) — those
  remain open.

## Metrics
- PESQ is intentionally omitted (see `research/questions.md` non-goals).
- ViSQOL is not yet integrated.
- STOI is speech-only by design (brief §17 — music needs separate metrics, not implemented).
- The hallucination-proxy panel currently implements only the ASR-transcript-divergence
  proxy end-to-end in the pipeline; spectral-energy-outside-reference and
  ensemble-disagreement proxies exist as separate metric functions
  (`src/soundrevive/metrics/`) but are not yet wired into `run_pipeline.py`'s per-sample loop.
- No proxy has been validated against a second, independent signal of "hallucination" —
  they are reported side by side, not fused, precisely because fusing unvalidated proxies
  into one score would itself be a fabricated metric (brief §15).

## What the first SMOKE-tier run actually found (2026-09-07/08)
- **Wow/flutter was removed from the default degradation chain.** It was in the chain
  initially (matching the brief's §7 example). With even LIGHT-tier depths, it introduced a
  continuously time-varying misalignment that the alignment step (a single constant lag,
  brief §18) cannot correct, and none of the v1 restoration methods attempt time-warp
  correction either — SI-SDR ended up dominated by uncorrected timing drift rather than by
  the noise/click/bandlimit damage the benchmark is meant to measure, to the point that it
  stopped differentiating between methods at all (passthrough and every restoration method
  scored within noise of each other, all deeply negative). `wow_flutter` remains implemented
  and available (`include_wow_flutter=True` in `sample_chain_config`); brief §39 already asks
  for it to be its own separate experiment with its own pitch-stability metrics, which is
  where it belongs given this finding.
- **A real, small-sample metric-disagreement result** (brief §74, of direct relevance to
  RQ2/H2): in the corrected run, the discriminative ML denoiser sometimes *increases* SI-SDR
  at SEVERE/EXTREME tiers relative to passthrough, while classical baselines barely move it —
  but the ML denoiser's log-spectral-distance is consistently far worse than every classical
  baseline (roughly 2-3x higher) at every tier. Time-domain SI-SDR and frequency-domain
  log-spectral-distance disagree about whether the ML method helped. At LIGHT/MODERATE tiers
  the ML denoiser actively *hurts* SI-SDR relative to doing nothing (passthrough), which is
  consistent with H1 (classical baselines not dominated at low severity) — see
  `results/benchmark.parquet`. This is a real observation from the actual data, not a
  fabricated one, but it rests on 2 clips/1 reader and must not be quoted as a generalizable
  finding — see `research/claims_registry.md` row C1 (status: SMOKE ONLY).
- **STFT/ISTFT bug found and fixed during this run**: a hand-rolled overlap-add
  reconstruction (`soundrevive/stft_utils.py`) was only an exact inverse for an *unmodified*
  spectrum; applying it to a magnitude-masked spectrum (exactly what spectral gating, Wiener
  filtering, and the ML denoiser all do) amplified reconstructed energy 3-14x in testing. It
  was replaced with a thin wrapper around `scipy.signal.stft/istft`, which handles modified
  spectra correctly (verified numerically). Documented here because it silently produced
  wrong numbers before `tests/test_baselines.py` caught the resulting energy increase.
- **Median declick's threshold had to be changed from a global to a local MAD estimate**: a
  single global noise-floor threshold over an entire clip flagged ~17% of samples as "clicks"
  on ordinary speech (fast transients in plosives/sibilants triggered it), making the
  restored signal *worse* than the clicked input. A locally-adaptive threshold (a second,
  wider median filter over the residual magnitude) fixed this; recall on synthetic clicks is
  still modest (~30%) at the threshold chosen to keep false positives low enough to help
  on net — see `src/soundrevive/baselines/declick.py` docstring.

## What the STANDARD-tier run actually found (2026-09-08)
This run (`src/soundrevive/pipeline/run_standard.py`, `results/standard/`) trained the
discriminative ML denoiser on 96 augmented pairs from 6 LibriSpeech speakers, then evaluated
all 5 methods on 24 clips from 6 *different*, never-trained-on speakers — a genuine
speaker-held-out generalization test. Ground truth for WER is the real human LibriSpeech
transcript, not a whisper pseudo-label (fixed a real bug in the process, see below).

- **The ML denoiser generalizes to unseen speakers on signal-fidelity metrics, and does so
  much better than the SMOKE-tier run suggested.** At MODERATE/SEVERE/EXTREME tiers it beats
  every classical baseline on both SI-SDR and log-spectral-distance, consistently across
  held-out speakers (checked per speaker×tier, not just averaged: it wins on SI-SDR in 12/12
  speaker×tier combinations at SEVERE+EXTREME). Unlike the SMOKE run, it does not hurt SI-SDR
  at LIGHT/MODERATE tiers either — plausibly because the STANDARD training set has 6x more
  distinct speakers/6x more training pairs than SMOKE's single-speaker corpus, though this is
  an unconfirmed explanation, not a controlled ablation.
- **The same ML denoiser does NOT generalize on content preservation, and this is the
  clearest single finding in the repo so far.** It has the worst ASR-WER of every method at
  every tier, and *increases* WER relative to doing nothing (the input's own WER) at every
  tier — while classical Wiener filtering frequently *decreases* WER (measurably improves
  transcribability) despite scoring lower on SI-SDR than the ML method. Checked per speaker,
  not just on average: the ML method has a worse WER-delta than Wiener in 10/12 speaker×tier
  combinations at SEVERE+EXTREME — the same 12/12 combinations where it wins on SI-SDR. This
  directly answers the brief's §16 question ("does restoration improve intelligibility
  without changing words?") for v1's methods: for Wiener filtering, mostly yes; for the ML
  denoiser, no — it suppresses noise energy effectively but the words underneath change more,
  not less, than if nothing had been done. STOI (the intelligibility *proxy* used here) moved
  in the *same* direction as SI-SDR for the ML method (both improved) — meaning STOI and the
  behavioral ASR-WER measure disagree with each other about whether intelligibility improved,
  which is itself worth flagging (brief §74) since STOI is being used here as a stand-in for
  exactly what WER measures directly.
- **A second real bug found and fixed in the process**: `word_error_rate`/`char_error_rate`
  (`src/soundrevive/metrics/intelligibility.py`) compared a raw Whisper hypothesis against the
  raw LibriSpeech reference transcript with no normalization. LibriSpeech transcripts are
  all-caps with no punctuation and spelled-out abbreviations ("MISTER QUILTER..."), while
  Whisper outputs mixed-case punctuated text ("Mr. Quilter..."). A perfect transcription
  scored WER=1.0 before fixing this — purely a formatting artifact, not a content error. Fixed
  by running both strings through Whisper's own `EnglishTextNormalizer` before scoring
  (the standard approach for evaluating Whisper against corpora like LibriSpeech). This did
  not affect the SMOKE run's numbers (both its hypothesis and reference came from Whisper
  itself, so casing/punctuation style already matched), but would have silently invalidated
  every WER number in this STANDARD run had it not been caught before the run — see
  `tests/test_metrics.py::test_word_error_rate_normalizes_case_and_punctuation`.

## Human data
- **No human listening-study data exists.** `results/listening_study.parquet` does not exist.
  Any future simulated-listener test of the study pipeline must be labeled `SIMULATED
  PIPELINE` in the UI and paper, never presented next to or interchangeably with real data.

## Compute / environment
- Development machine has limited free disk (~12–14 GB at project start); large pretrained
  model weights (e.g. big diffusion-based restoration checkpoints) are avoided by default.
  Any future generative baseline must document its weight size and hash before being added.
- All models trained/run on CPU in v1; no GPU-scale experiment has been run, so runtime
  numbers in `results/runtime.parquet` should not be extrapolated to GPU throughput.

## What "SMOKE" results in this repo do and do not support
They demonstrate that the pipeline mechanics (degrade → restore → align → measure → persist
with provenance) work end-to-end and are deterministic. They do **not** support any claim
about which method is scientifically better, by how much, or whether it would generalize.

## What "STANDARD" results in this repo do and do not support
The STANDARD run (12 speakers, real train/eval speaker split) supports a real, cross-speaker
directional finding — see "What the STANDARD-tier run actually found" above — but 6 held-out
speakers/24 clips is still too small to report an effect size, a confidence interval, or a
"method X beats method Y by N dB" headline number. It supports "this pattern held consistently
across every held-out speaker we tried," not "this pattern would hold at scale." Treat it as
a well-supported hypothesis for a larger run, not a finished benchmark result.

See `research/claims_registry.md` for the exact status of every number, in both tiers.
