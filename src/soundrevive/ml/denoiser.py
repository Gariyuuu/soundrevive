"""Discriminative ML restoration method (research/restoration_taxonomy.md §1): a small
STFT-magnitude-mask convolutional network, trained with an L1 mask-application loss against
paired clean/degraded audio. This is a standard, unremarkable architecture used here as a
benchmark *participant*, not a research contribution (research/novelty_memo.md).

SMOKE-tier scope: trained only on pairs synthesized from the 2-clip corpus via the
degradation engine itself (many degradation realizations of the same 2 recordings). This is
sufficient to verify the training/inference pipeline mechanics and is NOT a generalization
claim — see research/limitations.md.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from soundrevive.stft_utils import istft, stft

N_FFT = 512
HOP = 128
N_FREQ = N_FFT // 2 + 1


class STFTMaskDenoiser(nn.Module):
    """Conv1d stack over the frequency axis, per-frame, predicting a [0,1] soft mask applied
    to the noisy magnitude spectrogram. Small enough to train on CPU in seconds-to-minutes at
    SMOKE scale."""

    def __init__(self, n_freq: int = N_FREQ, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_freq, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, n_freq, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, mag_db: torch.Tensor) -> torch.Tensor:
        # mag_db: (batch, n_freq, n_frames)
        return self.net(mag_db)


def _mag_to_db_norm(mag: np.ndarray) -> np.ndarray:
    return np.log1p(mag).astype(np.float32)


def restore_with_denoiser(model: STFTMaskDenoiser, audio: np.ndarray) -> np.ndarray:
    model.eval()
    spec = stft(audio, N_FFT, HOP)
    mag = np.abs(spec)
    phase = np.angle(spec)
    feat = torch.from_numpy(_mag_to_db_norm(mag)).unsqueeze(0)
    with torch.no_grad():
        mask = model(feat).squeeze(0).numpy()
    out_mag = mag * mask
    out_spec = out_mag * np.exp(1j * phase)
    return istft(out_spec, N_FFT, HOP, length=len(audio))


def train_denoiser(clean_degraded_pairs: list[tuple[np.ndarray, np.ndarray]], epochs: int = 30,
                    lr: float = 1e-3, seed: int = 0) -> STFTMaskDenoiser:
    torch.manual_seed(seed)
    model = STFTMaskDenoiser()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    specs = []
    for clean, degraded in clean_degraded_pairs:
        clean_spec = stft(clean, N_FFT, HOP)
        deg_spec = stft(degraded, N_FFT, HOP)
        n = min(clean_spec.shape[1], deg_spec.shape[1])
        specs.append((clean_spec[:, :n], deg_spec[:, :n]))

    model.train()
    for _epoch in range(epochs):
        total_loss = 0.0
        for clean_spec, deg_spec in specs:
            deg_mag = np.abs(deg_spec)
            clean_mag = np.abs(clean_spec)
            feat = torch.from_numpy(_mag_to_db_norm(deg_mag)).unsqueeze(0)
            target = torch.from_numpy(clean_mag).unsqueeze(0)
            deg_mag_t = torch.from_numpy(deg_mag).unsqueeze(0)

            mask = model(feat)
            pred_mag = mask * deg_mag_t
            loss = torch.mean(torch.abs(pred_mag - target))

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
    return model


def save_denoiser(model: STFTMaskDenoiser, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_denoiser(path: str) -> STFTMaskDenoiser:
    model = STFTMaskDenoiser()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
