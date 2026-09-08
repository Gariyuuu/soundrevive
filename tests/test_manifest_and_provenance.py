import numpy as np

from soundrevive.io_utils import load_wav
from soundrevive.pipeline.manifest import clip_path, load_manifest
from soundrevive.provenance import config_hash, sha256_of_array, sha256_of_file


def test_manifest_clips_exist_and_hashes_match():
    manifest = load_manifest()
    for entry in manifest["clean_speech"]:
        path = clip_path(entry["clip_id"])
        assert path.exists(), f"missing {path}"
        assert sha256_of_file(str(path)) == entry["sha256_wav"], entry["clip_id"]


def test_manifest_declares_smoke_tier():
    manifest = load_manifest()
    assert manifest["tier"] == "SMOKE"


def test_loaded_audio_is_mono_16k_float32():
    path = clip_path("alice00_16k_clip")
    audio, sr = load_wav(str(path))
    assert sr == 16000
    assert audio.dtype == np.float32
    assert audio.ndim == 1


def test_config_hash_is_deterministic_and_order_sensitive():
    cfg_a = {"tier": "LIGHT", "seed": 1}
    cfg_b = {"seed": 1, "tier": "LIGHT"}  # different key order, same content
    assert config_hash(cfg_a) == config_hash(cfg_b)

    cfg_c = {"tier": "SEVERE", "seed": 1}
    assert config_hash(cfg_a) != config_hash(cfg_c)


def test_sha256_of_array_stable_across_calls():
    arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert sha256_of_array(arr) == sha256_of_array(arr.copy())
