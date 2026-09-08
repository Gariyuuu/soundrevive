# SoundRevive — Hypotheses

Pre-registered before the standard-tier run. Each is falsifiable against a named metric
pair in `results/`. Confirming a hypothesis is not the goal — reporting whichever way the
evidence actually falls is (see `research/claims_registry.md` and the "negative findings"
commitment in the project brief, §76).

## H1 — Classical baselines are not dominated
At LIGHT/MODERATE degradation tiers, at least one classical signal-processing baseline
(spectral gating, Wiener filtering, or bandpass+declick) will be within noise of the best
discriminative-ML method on SI-SDR, because these tiers mostly reintroduce noise/band-limit
damage that classical estimators already model well. We expect the gap to widen in favor of
ML only at SEVERE/EXTREME tiers. Test: paired bootstrap CI on SI-SDR, classical-best vs.
ML-best, per tier.

## H2 — Quality and fidelity diverge under generative restoration (flagship)
Generative/high-strength restoration will show a *negative* correlation between perceptual
quality proxies and reference fidelity once strength exceeds some threshold — i.e., turning
up "cleanness" trades away closeness to the reference. Discriminative methods will show this
divergence less, or not at all, across their operating range. Test: per-method Spearman
correlation between quality-axis and fidelity-axis metrics across the strength sweep (RQ3/§28
of the brief); compare sign/magnitude between generative and discriminative methods.

## H3 — Aggressiveness has a fidelity cliff, not a fidelity slope
We expect fidelity (SI-SDR vs. reference) to degrade non-linearly with restoration strength —
a plateau followed by a cliff — rather than degrading smoothly, once the method starts
replacing missing content instead of merely attenuating noise. Test: piecewise-linear vs.
linear fit comparison (AIC) on fidelity-vs-strength curves.

## H4 — Generative methods score worse on hallucination proxies at matched perceptual quality
Holding a no-reference perceptual-quality proxy roughly constant across methods, generative
restoration will score worse (more unsupported content) than discriminative restoration on
the hallucination proxies in `research/restoration_taxonomy.md`. Test: stratify samples into
quality-proxy bins; compare hallucination-proxy distributions within bin, generative vs.
discriminative.

## H5 — Compound degradation breaks isolated-degradation-trained methods more than tier alone predicts
A method's fidelity drop from an isolated degradation tier to a compound chain at a matched
"nominal severity" will exceed the drop predicted by summing single-degradation drops,
i.e. compound damage is not additive in its effect on restoration quality. Test: compare
observed compound-tier fidelity to a linear combination baseline; report residual.

## H6 — Ensemble disagreement correlates with true error, but imperfectly
Under Regime A (ground truth known), sample-level ensemble disagreement (spread of
restoration outputs across methods) will positively correlate with true reconstruction
error, but the correlation will be well below 1.0 — i.e. disagreement is a weak, not
strong, proxy for reliability. Test: Pearson/Spearman correlation, disagreement-metric vs.
ground-truth error, with bootstrap CI reported (not just the point estimate).

## H7 — Dropout inpainting is the single highest-hallucination-risk degradation type
Among all degradation types tested, sample-dropout (full-signal gaps) will produce the
largest average hallucination-proxy scores under generative restoration, because there is
no attenuated original signal left to anchor reconstruction (unlike noise/band-limiting,
where the underlying signal is degraded but present). Test: compare hallucination-proxy
distributions across degradation types, generative methods only.

## Status
All hypotheses are currently **UNTESTED** — see `results/release.json` for which have a
completed, checked-in test as of the current pipeline run. The first (smoke-tier) pipeline
run in this repo exercises the *mechanics* of H1 and H6 only, on 2 source clips; it is not
sufficient evidence to confirm or reject any hypothesis (see `research/limitations.md`).
