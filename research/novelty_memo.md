# SoundRevive — Novelty Memo

## What already exists (well-established, not novel)
- Speech enhancement / denoising benchmarks (DNS Challenge and similar) evaluate
  discriminative models against synthetic noise + clean speech, using SI-SDR/PESQ/STOI.
  This is a mature, standard evaluation paradigm.
- Audio super-resolution / bandwidth extension has its own benchmark literature, typically
  scored on spectral/fidelity metrics alone.
- Perceptual audio-quality metrics (PESQ, STOI, ViSQOL) are individually well studied,
  each validated against human ratings in its own original domain (mostly telephony/VoIP
  speech, not historical-recording restoration).

None of the above, on their own, ask whether "sounds better" and "is more faithful to the
original" are the same axis — most existing work reports a single quality-style metric and
treats improvement on it as the finding.

## What is claimed as this project's contribution
1. **A fidelity/quality two-axis frontier specifically for historical-style degradation**,
   rather than telephony-style stationary noise — i.e. evaluating restoration methods against
   *both* a reference-fidelity axis and a perceptual/no-reference-quality axis, and reporting
   where methods trade one for the other (RQ2/H2), rather than reporting a single leaderboard
   number.
2. **An explicit hallucination-proxy panel** (spectral, ASR-divergence, ensemble-disagreement)
   applied specifically to the historical-restoration setting, where "plausible but wrong"
   reconstruction is a live risk (bandwidth extension, dropout inpainting) — as opposed to
   most enhancement benchmarks, which don't measure content preservation at all.
3. **A two-regime design that refuses to conflate synthetic-ground-truth results with
   genuine-historical no-reference results** — many restoration demos implicitly suggest that
   good synthetic-benchmark numbers justify trusting outputs on real archival material; this
   project keeps the two regimes structurally separate (separate result files, separate
   claims) specifically to avoid that conflation.
4. **A strength/aggressiveness sweep as a first-class experiment** rather than a fixed
   operating point, to characterize *when* a method crosses from correction into invention
   (RQ3/H3), which most single-operating-point benchmarks don't expose.

## What is explicitly NOT claimed as novel
- The individual metrics (SI-SDR, STOI, WER-based content checks) are standard tools, applied
  here, not invented here.
- The individual degradation primitives (bandlimit, hiss, clicks, wow/flutter) are standard
  audio-engineering constructs; the contribution is their historically-motivated composition
  and tiering, not the primitives themselves.
- No claim is made that this project discovers a new restoration architecture; v1's ML
  method is a standard STFT-mask discriminative model used as a benchmark participant, not a
  research contribution in itself. A generative-restoration participant is currently absent
  (see `research/limitations.md`) — the fidelity/quality divergence claim (H2) cannot be
  fully evaluated without one, and the project is explicit about that gap rather than
  simulating it with the discriminative model.

## Risk to the novelty claim
If a generative-restoration baseline is never added, the "cleaner is not closer" framing
(the working paper title) is under-supported — a discriminative-only benchmark can show
noise-vs-fidelity tradeoffs but cannot show the sharper generative hallucination story the
brief centers on. This is tracked as the top open item in `docs/HANDOFF.md`.
