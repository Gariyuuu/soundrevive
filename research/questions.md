# SoundRevive — Research Questions

Central tension the project is built to answer:

> Restoration algorithms can make historical audio sound cleaner, but how do we know
> they preserved the original signal rather than hallucinating plausible modern audio?

All seven questions below are pre-registered before the standard-tier benchmark is run.
Each maps to concrete artifacts in `results/` and to a section of `research/claims_registry.md`.
None may be answered by a hand-typed number — only by a script whose output is checked in.

## RQ1 — Recovery under known degradation
Which restoration methods best recover known clean audio after historically plausible,
synthetic degradation (Regime A)? Measured by SI-SDR / SDR / SNR-improvement / log-spectral
distance against ground truth, per method, per degradation tier.

## RQ2 — Quality vs. fidelity divergence
Do methods that maximize perceptual quality (STOI, no-reference proxies, human preference)
also maximize signal fidelity (SI-SDR, spectral distance)? Or does the ranking invert?
This is the project's flagship experiment (see `research/hypotheses.md` H2).

## RQ3 — Aggressiveness ceiling
How much restoration can be applied before content (not just noise) begins to change?
Studied via a strength/temperature sweep on any restoration method that exposes a
continuous aggressiveness parameter, tracking fidelity and a content-preservation proxy
(WER for speech) as a joint function of strength.

## RQ4 — Generative hallucination propensity
Are generative restoration methods more likely than discriminative ones to introduce
spectral or phonetic content unsupported by the reference? Compared via the hallucination
proxies in `research/restoration_taxonomy.md` §Hallucination Proxies, split by method
category (signal processing / discriminative ML / generative ML).

## RQ5 — Cross-condition generalization
How well do methods trained/tuned on one degradation distribution generalize to unseen
noise types, unseen bandwidth limits, unseen severities, and unseen recording classes?

## RQ6 — Disagreement as an uncertainty signal
Can ensemble disagreement across restoration methods flag regions where reconstruction is
unreliable, in the absence of ground truth (Regime B)? Validated against Regime A, where
ground truth lets us check whether high-disagreement regions really do have higher error.

## RQ7 — Do objective metrics predict human preference?
Do the objective fidelity/quality metrics used here actually predict which restoration a
human listener prefers, and prefers as *closer to the reference*? Answerable only once
`results/listening_study.parquet` contains real paired-comparison data — see
`research/limitations.md` for current status (as of this writing: **NO HUMAN DATA**,
infrastructure only).

## Non-goals (explicitly out of scope for v1)
- Music and environmental-audio restoration (speech is the tractable flagship case per the
  project brief; music/environmental hooks are stubbed, not scored, in v1).
- PESQ (ITU-T P.862 licensing terms are restrictive for redistribution of implementations in
  some jurisdictions and require care; v1 relies on SI-SDR/SDR/STOI/spectral-distance and
  documents PESQ's omission rather than shipping an unlicensed/unverified implementation).
- Any claim about a *specific* named historical recording's authenticity beyond what its
  archive's own catalog record states.
