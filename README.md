# SoundRevive

A research-grade benchmark and lab for ML-based restoration of historical degraded audio,
built around one question: restoration can make historical audio sound cleaner, but does
that mean it preserved the original signal, or did it hallucinate plausible modern audio?

Start here:
- `docs/HANDOFF.md` — current status, what's done, what isn't, exact next steps.
- `research/questions.md`, `research/hypotheses.md` — what this project is trying to answer.
- `research/claims_registry.md` — every number this repo can currently back up, and its tier.
- `research/limitations.md` — what the SMOKE-tier results do and do not support.

## Quickstart

```
make setup      # python venv + deps
make test       # pytest
make benchmark  # run the SMOKE-tier pipeline (degrade -> restore -> measure -> results/)
```

See `Makefile` for the full target list (some, e.g. `figures`/`paper`/`site`, are not
implemented yet — they fail loudly rather than silently no-op).
