.PHONY: setup data-smoke benchmark-standard degrade restore-smoke benchmark hallucination analyze figures paper site site-data site-build test reproduce

VENV := .venv/bin

setup:
	python3.11 -m venv .venv
	$(VENV)/pip install -q --upgrade pip
	$(VENV)/pip install -e .
	$(VENV)/pip install -r requirements.txt

# Both corpora are already committed — nothing to download.
# SMOKE: data/raw/clean_speech/ (2 CC0 LibriVox clips).
# STANDARD: data/raw/librispeech_dev_clean/ (48-clip, 12-speaker CC BY 4.0 subset of
# LibriSpeech dev-clean, with a real train_pool/eval_pool speaker split — see
# data/manifest.json's librispeech_speaker_split and research/data_provenance.md).
data-smoke:
	$(VENV)/python -m pytest tests/test_manifest_and_provenance.py -q

benchmark-standard:
	$(VENV)/python -m soundrevive.pipeline.run_standard

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

# Site (brief §41): currently one home page reporting the STANDARD-tier headline finding —
# see site/README.md. The other routes (/benchmark, /listen, /degradations, ...) don't exist
# yet, see docs/HANDOFF.md.
site-data:
	$(VENV)/python scripts/generate_site_summary.py

site:
	cd site && npm run dev

site-build:
	cd site && npm run build

test:
	$(VENV)/python -m pytest -q

reproduce: test benchmark analyze
