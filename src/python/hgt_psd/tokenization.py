"""Hierarchical Genomic Tokenization: 250-kb coarse -> 10-kb fine bins."""

from __future__ import annotations

import numpy as np

ALPHABET = "ACGT"
SYM = {c: i for i, c in enumerate(ALPHABET)}


class HierarchicalGenomicTokenizer:
    def __init__(self, coarse_bin_bp: int = 250_000, fine_bin_bp: int = 10_000):
        self.coarse_bin_bp = coarse_bin_bp
        self.fine_bin_bp = fine_bin_bp
        if coarse_bin_bp % fine_bin_bp != 0:
            raise ValueError("coarse_bin_bp must be divisible by fine_bin_bp")
        self.fine_per_coarse = coarse_bin_bp // fine_bin_bp

    def one_hot(self, seq: str) -> np.ndarray:
        arr = np.zeros((len(seq), 4), dtype=np.float32)
        for i, ch in enumerate(seq.upper()):
            if ch in SYM:
                arr[i, SYM[ch]] = 1.0
            else:
                arr[i] = 0.25
        return arr

    def tokenize_region(self, seq: str, n_bins: int) -> np.ndarray:
        """
        Map sequence to n_bins tokens (one-hot averaged per bin).
        Returns (n_bins, 4).
        """
        seq = seq.upper()
        L = len(seq)
        bin_size = max(L // n_bins, 1)
        tokens = np.zeros((n_bins, 4), dtype=np.float32)
        for b in range(n_bins):
            start = b * bin_size
            end = min(L, (b + 1) * bin_size) if b < n_bins - 1 else L
            if start >= end:
                tokens[b] = 0.25
                continue
            chunk = self.one_hot(seq[start:end])
            tokens[b] = chunk.mean(axis=0)
        return tokens

    def hierarchical_grid(self, seq: str, n_coarse: int, n_fine: int) -> dict:
        """Coarse 250-kb grid pooled to n_coarse; fine 10-kb grid to n_fine."""
        coarse_tokens = self.tokenize_region(seq, n_coarse)
        fine_tokens = self.tokenize_region(seq, n_fine)
        return {
            "coarse": coarse_tokens,
            "fine": fine_tokens,
            "coarse_bin_bp": self.coarse_bin_bp,
            "fine_bin_bp": self.fine_bin_bp,
        }
