"""Every generated artifact carries enough metadata to answer 'exactly how was this made'
without re-running anything (brief §63)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_of_array(arr) -> str:
    import numpy as np

    a = np.ascontiguousarray(arr, dtype=np.float32)
    return hashlib.sha256(a.tobytes()).hexdigest()


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def git_sha(short: bool = True) -> str:
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        out = subprocess.run(args, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2], check=False)
        sha = out.stdout.strip()
        return sha if sha else "UNCOMMITTED"
    except OSError:
        return "UNKNOWN"
