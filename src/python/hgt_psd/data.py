"""Hi-C matrix loading and synthetic demo generators."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def synthesize_psd_contact_matrix(
    n: int, rank: int = 8, seed: int = 42, noise_scale: float = 0.05
) -> np.ndarray:
    """Generate KR-like PSD contact matrix with distance decay."""
    rng = np.random.default_rng(seed)
    freqs = np.linspace(0.1, 2.0, rank)
    i = np.arange(n)
    Psi = np.cos(i[:, None] * freqs[None, :])
    alpha = rng.dirichlet(np.ones(rank))
    Phi = Psi * np.sqrt(alpha)
    Y = Phi @ Phi.T
    # distance decay
    dist = np.abs(i[:, None] - i[None, :])
    Y = Y * np.exp(-dist / (n / 3))
    Y += noise_scale * np.diag(rng.random(n))
    Y = (Y + Y.T) / 2
    return Y.astype(np.float32)


def synthesize_sequence(length: int, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    bases = "ACGT"
    return "".join(rng.choice(list(bases)) for _ in range(length))


def load_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    raise ValueError(f"Unsupported matrix format: {path}")


def save_matrix(path: Path, mat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mat)


def prepare_batch_numpy(seq_tokens: np.ndarray, contact: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return seq_tokens, contact


def load_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))
