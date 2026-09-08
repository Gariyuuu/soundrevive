"""Load the clean-reference data manifest (data/manifest.json) — the single source of truth
for what audio exists, where it came from, and under what license (brief §4, §63)."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_manifest() -> dict:
    with open(REPO_ROOT / "data" / "manifest.json") as f:
        return json.load(f)


def clip_path(clip_id: str) -> Path:
    manifest = load_manifest()
    for entry in manifest["clean_speech"]:
        if entry["clip_id"] == clip_id:
            return REPO_ROOT / entry["path"]
    raise KeyError(f"no clip_id {clip_id!r} in manifest")
