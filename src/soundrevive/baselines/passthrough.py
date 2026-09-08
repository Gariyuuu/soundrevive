"""The mandatory 'no processing' control (brief §10) — every other method must beat this to
justify existing."""
import numpy as np


def passthrough(audio: np.ndarray, sr: int) -> np.ndarray:
    return audio.copy()
