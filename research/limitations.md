# SoundRevive — Limitations (living document)

Updated as of the first pipeline run. This file is a commitment device: it must be read and
updated before any claim is promoted from `SMOKE ONLY` to `STANDARD` in
`research/claims_registry.md`.

## Data
- **Clean reference corpus is currently 2 clips (~40s total), single speaker, single
  recording session** (public-domain LibriVox recordings of "New Adventures of Alice" read
  by John Rae, CC0, verified via the Internet Archive metadata API — see
  `research/data_provenance.md`). This is a SMOKE-tier corpus for pipeline verification only.
  It cannot support any generalization, ranking-stability, or fidelity-vs-quality claim.
  A STANDARD-tier run requires a proper clean speech corpus (multi-speaker, studio-quality).
  `research/data_provenance.md` (separate research pass, completed) recommends **LJSpeech**
  (verified true public domain) as the primary STANDARD-tier candidate, with **LibriSpeech**
  (CC BY 4.0, verified) as a second choice for speaker/scale diversity. It also found no
  large, unambiguously rights-clear corpus of genuine early-20th-century *speech* (as opposed
  to music) for Regime B — see that file's "Open gap" note; Regime B speech will likely be
  small-volume UCSB pre-1923 cylinder material plus Library-of-Congress items usable only for
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
  discriminative ML method (STFT-mask conv denoiser) trained only on augmented pairs derived
  from the same 2-clip corpus. **No generative-restoration method exists yet.** This blocks
  full evaluation of RQ2/RQ4/H2/H4 — the project's flagship "cleaner is not closer" framing
  needs a method that can trade fidelity for perceived quality via a strength parameter, and
  none is implemented yet.
- The discriminative model's "generalization" has not been tested against held-out speakers,
  recording conditions, or degradation types (RQ5) — it has only been evaluated on
  degradation configs drawn from the same distribution it was exposed to during training.

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
See `research/claims_registry.md` for the exact status of every number.
