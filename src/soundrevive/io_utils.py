"""Audio I/O helpers. Wraps soundfile so the rest of the codebase never touches file formats
directly, and always works in float32, [-1, 1]-scaled, mono-or-explicit-channel arrays."""
from __future__ import annotations

import numpy as np
import soundfile as sf


def load_wav(path: str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    if target_sr is not None and sr != target_sr:
        audio = resample(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(path, audio, sr, subtype="PCM_16")


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    import torch
    import torchaudio

    wav = torch.from_numpy(audio).unsqueeze(0)
    out = torchaudio.functional.resample(wav, sr_in, sr_out)
    return out.squeeze(0).numpy().astype(np.float32)
