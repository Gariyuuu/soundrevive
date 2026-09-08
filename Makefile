.PHONY: setup data-smoke degrade restore-smoke benchmark hallucination analyze figures paper site test reproduce

VENV := .venv/bin

setup:
	python3.11 -m venv .venv
	$(VENV)/pip install -q --upgrade pip
	$(VENV)/pip install -e .
	$(VENV)/pip install -r requirements.txt

# The SMOKE-tier clean corpus is already committed under data/raw/clean_speech/ (2 CC0
# LibriVox clips, see data/manifest.json) — nothing to download for the smoke pipeline.
# A STANDARD-tier `data-standard` target (fetching LJSpeech/LibriSpeech per
# research/data_provenance.md) is not implemented yet; see docs/HANDOFF.md.
data-smoke:
	$(VENV)/python -m pytest tests/test_manifest_and_provenance.py -q

degrade:
	$(VENV)/python -c "from soundrevive.pipeline.run_pipeline import build_training_pairs; print(len(build_training_pairs(16000)), 'degraded training pairs built (in-memory smoke check)')"

# There is no separate "restore-smoke" step — restoration happens inside run_pipeline.py
# alongside measurement, so this target just runs the same smoke pipeline as `benchmark`.
restore-smoke: benchmark

benchmark:
	$(VENV)/python -m soundrevive.pipeline.run_pipeline

# Hallucination-proxy results (ASR-WER leg) are produced as part of `benchmark`
# (results/hallucination.parquet). Spectral-energy and ensemble-disagreement proxies exist
# as functions but are not yet wired into the pipeline runner — see research/limitations.md.
hallucination: benchmark

analyze:
	$(VENV)/python -c "import pandas as pd; df = pd.read_parquet('results/benchmark.parquet'); print(df.groupby(['tier','method'])[['si_sdr_db','stoi','log_spectral_distance_db']].mean())"

# Figure generation is not implemented yet (brief §49) — see docs/HANDOFF.md.
figures:
	@echo "not implemented yet — see docs/HANDOFF.md"
	@exit 1

# Paper generation is not implemented yet (brief §46) — see docs/HANDOFF.md.
paper:
	@echo "not implemented yet — see docs/HANDOFF.md"
	@exit 1

# The Next.js site is not implemented yet (brief §41) — see docs/HANDOFF.md.
site:
	@echo "not implemented yet — see docs/HANDOFF.md"
	@exit 1

test:
	$(VENV)/python -m pytest -q

reproduce: test benchmark analyze
