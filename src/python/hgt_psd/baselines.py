"""Baseline model stubs for ablation comparison."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankFactorized(nn.Module):
    """Unconstrained low-rank factorization (not guaranteed PSD)."""

    def __init__(self, d_model: int = 512, K: int = 32, n: int = 25):
        super().__init__()
        self.U = nn.Linear(d_model, n * K)
        self.input_proj = nn.Linear(4, d_model)

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(seq_tokens).mean(1)
        n = int(seq_tokens.shape[1]) if seq_tokens.shape[1] else 25
        K = 32
        U = self.U(h).view(-1, n, K)
        return torch.einsum("bnk,bmk->bnm", U, U)


class DeepCStub(nn.Module):
    """Lightweight convolutional stub (DeepC-style)."""

    def __init__(self, n: int = 25, d_model: int = 128):
        super().__init__()
        self.n = n
        self.conv = nn.Sequential(
            nn.Conv1d(4, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(d_model, n * n)

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        x = seq_tokens.transpose(1, 2)
        h = self.conv(x).mean(-1)
        out = self.head(h).view(-1, self.n, self.n)
        return (out + out.transpose(-1, -2)) / 2


class AkitaStub(nn.Module):
    """Minimal dilated tower stub (Akita-style)."""

    def __init__(self, n: int = 25, channels: int = 64):
        super().__init__()
        self.n = n
        layers = []
        ch_in = 4
        for dilation in [1, 2, 4, 8]:
            layers += [
                nn.Conv1d(ch_in, channels, kernel_size=3, padding=dilation, dilation=dilation),
                nn.ReLU(),
            ]
            ch_in = channels
        self.tower = nn.Sequential(*layers)
        self.head = nn.Linear(channels, n * n)

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        x = seq_tokens.transpose(1, 2)
        h = self.tower(x).mean(-1)
        out = self.head(h).view(-1, self.n, self.n)
        return (out + out.transpose(-1, -2)) / 2
