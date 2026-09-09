"""Speech content-preservation metrics (brief §16). STOI is a reference-based intelligibility
proxy; the ASR-WER path is also used as one leg of the hallucination-proxy panel
(research/restoration_taxonomy.md §4) since it isolates content changes introduced by
restoration from pre-existing degradation damage, by always comparing against the clean
reference transcript rather than the degraded one."""
from __future__ import annotations

import numpy as np
from pystoi import stoi as _stoi

_WHISPER_MODEL = None
_TEXT_NORMALIZER = None


def _normalize(text: str) -> str:
    """Case/punctuation/number-format normalization before WER/CER (via Whisper's own
    EnglishTextNormalizer). Without this, comparing a Whisper hypothesis against a
    ground-truth transcript in a different convention (e.g. LibriSpeech's all-caps,
    no-punctuation, spelled-out-abbreviation style) inflates WER to ~1.0 on a perfect
    transcription purely from formatting, not content — verified empirically: "MISTER
    QUILTER IS..." vs. "Mr. Quilter is..." scores WER=1.0 unnormalized, 0.0 normalized."""
    global _TEXT_NORMALIZER
    if _TEXT_NORMALIZER is None:
        from whisper.normalizers import EnglishTextNormalizer

        _TEXT_NORMALIZER = EnglishTextNormalizer()
    return _TEXT_NORMALIZER(text)


def stoi_score(estimate: np.ndarray, reference: np.ndarray, sr: int) -> float:
    n = min(len(estimate), len(reference))
    return float(_stoi(reference[:n], estimate[:n], sr, extended=False))


def _load_whisper(model_size: str = "tiny.en"):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        import whisper

        _WHISPER_MODEL = whisper.load_model(model_size)
    return _WHISPER_MODEL


def transcribe(audio: np.ndarray, sr: int, model_size: str = "tiny.en") -> str:
    import torch
    import torchaudio

    model = _load_whisper(model_size)
    if sr != 16000:
        wav = torch.from_numpy(audio).unsqueeze(0)
        audio = torchaudio.functional.resample(wav, sr, 16000).squeeze(0).numpy()
    result = model.transcribe(audio.astype(np.float32), fp16=False, language="en")
    return result["text"].strip()


def word_error_rate(hypothesis: str, reference: str) -> float:
    import jiwer

    if not reference.strip():
        return float("nan")
    ref_norm, hyp_norm = _normalize(reference), _normalize(hypothesis)
    if not ref_norm.strip():
        return float("nan")
    return float(jiwer.wer(ref_norm, hyp_norm))


def char_error_rate(hypothesis: str, reference: str) -> float:
    import jiwer

    if not reference.strip():
        return float("nan")
    ref_norm, hyp_norm = _normalize(reference), _normalize(hypothesis)
    if not ref_norm.strip():
        return float("nan")
    return float(jiwer.cer(ref_norm, hyp_norm))
