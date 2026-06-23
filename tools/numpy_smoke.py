#!/usr/bin/env python3
"""NumPy-only smoke test (no PyTorch) — PSD proposition + stratified loss math."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))

from hgt_psd.data import (  # noqa: E402
    load_manifest,
    save_matrix,
    synthesize_psd_contact_matrix,
    synthesize_sequence,
)
from hgt_psd.tokenization import HierarchicalGenomicTokenizer  # noqa: E402


def structured_psd_numpy(tokens: np.ndarray, K: int = 32, n: int = 25) -> np.ndarray:
    """Numpy PSD synthesis mirroring StructuredPSDModel."""
    d_model = tokens.shape[1] * 4
    W_alpha = np.random.randn(d_model, K) * 0.01
    W_sigma = np.random.randn(d_model, n) * 0.01
    x = tokens.mean(axis=0).reshape(1, -1)
    if x.shape[1] < d_model:
        x = np.pad(x, ((0, 0), (0, d_model - x.shape[1])), constant_values=0.25)
    alpha = np.exp(x @ W_alpha)
    alpha = alpha / alpha.sum()
    sigma2 = np.log1p(np.exp(x @ W_sigma))
    freqs = np.linspace(0.1, 2.0, K)
    i = np.arange(n)
    Psi = np.cos(i[:, None] * freqs[None, :])
    Phi = Psi * np.sqrt(alpha + 1e-8)
    Y = Phi @ Phi.T + np.diag(sigma2.flatten())
    return Y.astype(np.float32)


def stratified_loss_numpy(Y: np.ndarray, Y_hat: np.ndarray, strata: dict, eps: float = 1e-6) -> float:
    n = Y.shape[0]
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :])
    total = 0.0
    for bounds in strata.values():
        lo = bounds.get("min_dist", 0)
        hi = bounds.get("max_dist")
        mask = dist >= lo if hi is None else (dist >= lo) & (dist <= hi)
        diff = (Y - Y_hat) * mask
        target = Y * mask
        num = (diff ** 2).sum()
        denom = target.var() + eps
        total += num / denom
    return float(total)


def main() -> None:
    manifest = load_manifest(ROOT / "manifests" / "gm12878_hic.json")
    n = manifest["model_defaults"]["n_bins"]
    seed = manifest["model_defaults"]["seed"]

    OUT = ROOT / "data" / "matrices"
    OUT.mkdir(parents=True, exist_ok=True)
    fetch_path = ROOT / "raw_outputs" / "fetch_manifest.json"
    if fetch_path.exists() and not json.loads(fetch_path.read_text()).get("demo_mode", True):
        fetch_meta = json.loads(fetch_path.read_text())
        contact = np.load(ROOT / fetch_meta["matrix_path"])
        tokens = np.load(ROOT / fetch_meta["tokens_path"])
        empirical_note = fetch_meta.get("fetch_mode", "public")
    else:
        contact = synthesize_psd_contact_matrix(n=n, rank=8, seed=seed)
        save_matrix(OUT / "gm12878_contact.npy", contact)
        seq = synthesize_sequence(250_000, seed=seed)
        (OUT / "gm12878_sequence.txt").write_text(seq, encoding="utf-8")
        tok = HierarchicalGenomicTokenizer()
        tokens = tok.tokenize_region(seq, n)
        np.save(OUT / "gm12878_tokens.npy", tokens)
        empirical_note = "demo_fallback"

    np.random.seed(seed)
    Y_hat = structured_psd_numpy(tokens, K=manifest["model_defaults"]["K"], n=n)
    evals = np.linalg.eigvalsh(Y_hat)
    psd_ok = evals.min() >= -1e-6
    rel_err = np.linalg.norm(contact - Y_hat, "fro") / np.linalg.norm(contact, "fro")
    loss = stratified_loss_numpy(contact, Y_hat, manifest["loss_strata"])

    audit = {
        "mode": "numpy_smoke",
        "empirical_fetch_mode": empirical_note,
        "proposition1_psd_valid": bool(psd_ok),
        "min_eigenvalue": float(evals.min()),
        "frobenius_rel_error": float(rel_err),
        "stratified_loss": loss,
        "note": "Full PyTorch training requires MSVC redist on Windows or Docker/Linux",
    }
    out = ROOT / "raw_outputs" / "audit_ledger.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if not fetch_path.exists() or json.loads(fetch_path.read_text()).get("demo_mode", True):
        fetch_meta = {
            "demo_mode": True,
            "matrix_path": "data/matrices/gm12878_contact.npy",
            "tokens_path": "data/matrices/gm12878_tokens.npy",
            "numpy_smoke": True,
        }
        fetch_path.write_text(json.dumps(fetch_meta, indent=2))

    print("=== NumPy Smoke Test ===")
    print(f"PSD valid: {psd_ok}")
    print(f"Min eigenvalue: {evals.min():.6f}")
    print(f"Rel Frobenius error: {rel_err:.4f}")
    print(f"Stratified loss: {loss:.4f}")
    print(f"Audit -> {out}")


if __name__ == "__main__":
    main()
