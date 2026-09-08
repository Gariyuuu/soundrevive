# SoundRevive — Restoration & Degradation Taxonomy

## Status of this document
The parameter ranges below are **engineering approximations** grounded in well-documented,
generally-known characteristics of historical recording/playback chains (shellac 78rpm discs,
wax cylinders, early optical film sound, AM radio bandwidth, early magnetic tape). They are
placeholders for literature-verified ranges: `research/literature_matrix.csv` (populated by a
separate research pass, see `docs/HANDOFF.md` for status) should be cross-checked against this
file before the STANDARD-tier benchmark run, and any range not corroborated by a citable source
must be labeled "engineering approximation, uncited" in the degradation config's own metadata
field, not silently presented as historically validated. This file itself is not a citation.

## 1. Restoration method categories

| Category | Definition | v1 members |
|---|---|---|
| SIGNAL PROCESSING | Fixed-form, non-learned estimators (filters, spectral subtraction) | passthrough (no-op control), bandpass EQ, spectral gating, Wiener filter, median-filter declick |
| DISCRIMINATIVE ML | Learned mapping degraded→clean, trained with a fidelity-style loss (L1/L2/STFT) against paired ground truth | `soundrevive.ml.denoiser` (STFT-mask conv model) |
| GENERATIVE ML | Learned model that can synthesize plausible content not directly inferable from the degraded signal (e.g. diffusion, strength/temperature-controllable) | not yet implemented in v1 (see `research/limitations.md`) — categories 15/28/71 of the brief require this to test H2/H4 and are currently open work |

Each result row in `results/benchmark.parquet` carries a `method_category` column so all
downstream analysis (RQ2, RQ4, H1–H4) can be sliced by category without re-deriving it.

## 2. Degradation primitives (v1 implemented set)

Implemented in `src/soundrevive/degradation/primitives.py`. Each primitive takes a `seed` and
is deterministic given (input, seed, params) — verified by
`tests/test_degradation_determinism.py`.

| Primitive | Historical motivation (approximate, see status note above) | Key params |
|---|---|---|
| `bandlimit` | Shellac/cylinder playback and early telephone/radio chains roll off below ~100–300 Hz and above ~3.5–8 kHz depending on era/format; modeled as cascaded Butterworth high/low-pass | `low_hz`, `high_hz`, `order` |
| `surface_noise` (hiss) | Shellac surface noise and tape hiss, approximated as colored (pink-leaning) stationary noise rather than white, since historical noise floors are not flat-spectrum | `snr_db`, `color_alpha` |
| `hum` | Mains-frequency electrical hum from playback/recording electronics | `freq_hz` (50/60), `n_harmonics`, `amp_db` |
| `clicks_pops` | Shellac disc surface damage / cylinder wear producing short (~1–5 ms) broadband impulses at random locations | `rate_per_sec`, `amplitude_db`, `width_ms` |
| `crackle` | Denser, lower-amplitude impulse train than discrete clicks; continuous surface noise texture | `density`, `amplitude_db` |
| `wow_flutter` | Mechanical speed instability in disc/cylinder/tape transports; modeled as slow (wow, ~0.5–6 Hz) and fast (flutter, ~6–20 Hz) sinusoidal time-warping via resampling | `wow_hz`, `wow_depth`, `flutter_hz`, `flutter_depth` |
| `nonlinear_distortion` | Overload distortion from mechanical groove tracing limits / early amplifier clipping | `drive_db`, `curve` (`tanh`/`hard_clip`) |
| `quantization` | Reduced effective bit depth from early digitization of analog transfers | `bits` |
| `dropout` | Missing/skipped audio segments (damaged media, transfer errors) | `n_dropouts`, `duration_ms_range` |
| `reverb` | Room/venue acoustics of the original recording space (not itself "damage," but a real, non-artifact-free condition to restore around) | `rt60_s`, `wet_level` |

Composable chains (config-driven, see `src/soundrevive/degradation/chains.py`) combine an
ordered subset of the above with a fixed random seed; the *exact* chain (ordered primitive
list + params + seed) is stored alongside every degraded file so no degradation is ever
regenerated implicitly with different results.

## 3. Difficulty tiers

Quantitative definitions (v1; subject to revision once literature-verified ranges land):

| Tier | Target broadband SNR after primitives (excl. dropout/reverb) | Bandlimit (typical) | Click rate | Wow depth |
|---|---|---|---|---|
| LIGHT | 25–35 dB | 100–5000 Hz | 0.5–2/s | ≤0.1% |
| MODERATE | 15–25 dB | 150–4000 Hz | 2–6/s | 0.1–0.3% |
| SEVERE | 5–15 dB | 200–3200 Hz | 6–15/s | 0.3–0.6% |
| EXTREME | −5–5 dB | 300–2500 Hz | 15–30/s | 0.6–1.2% |

Exact per-tier parameter distributions are defined in `src/soundrevive/degradation/tiers.py`,
not hand-typed here, so the table above is documentation, not the source of truth.

## 4. Hallucination proxies (v1)

No single "hallucination score" is treated as validated; the project reports several
independent proxies and their *disagreement* rather than collapsing them into one number
(brief §15 explicitly warns against inventing an unvalidated single score):

1. **Out-of-reference spectral energy** — energy in restored output's spectrogram bins that
   exceeds the reference's energy at that time-frequency cell by more than a noise-floor
   margin, after alignment.
2. **ASR transcript divergence (speech only)** — WER/CER between an ASR transcript of the
   restored output and of the clean reference (not the degraded input), isolating content
   changes introduced by restoration itself from pre-existing degradation damage.
3. **Ensemble disagreement** — spread of multiple restoration methods' outputs at a given
   time index (RQ6/§23); a proxy for uncertainty, validated against true error only where
   ground truth exists (Regime A).

These proxies are reported side by side per sample/method in `results/hallucination.parquet`;
`research/claims_registry.md` tracks which cross-proxy agreement claims are actually
supported by the data versus merely plausible.

## 5. RESTORED / ENHANCED / RECONSTRUCTED / GENERATED — labeling rule

Every processed audio artifact and every UI display of one carries exactly one label:

- **RESTORED** — output where the method only attenuates/corrects estimated damage on
  segments where the original signal is still present in some form (e.g. denoising,
  declicking, bandpass correction). No new content is synthesized.
- **ENHANCED** — output altered for subjective pleasantness (e.g. loudness normalization,
  EQ shaping) beyond damage correction, still without inventing content.
- **RECONSTRUCTED** — output where a missing/destroyed segment is filled using strong priors
  from the *surrounding signal itself* (e.g. short click/dropout interpolation) — plausible,
  bounded-uncertainty inference, not free synthesis.
- **GENERATED** — output where a model synthesizes content with no direct signal support
  (e.g. generative bandwidth extension inventing harmonics, generative inpainting of a long
  gap). Must be visually/audibly flagged wherever shown (brief §70).

This mapping is implemented as a required, non-optional field (`content_label`) on every
method's output metadata in `src/soundrevive/pipeline/run_pipeline.py` — a method without an
assigned label cannot be scored.
