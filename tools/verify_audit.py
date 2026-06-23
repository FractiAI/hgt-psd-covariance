#!/usr/bin/env python3
"""Verify PSD validity, rank ablation, and audit ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))

from hgt_psd.data import load_manifest  # noqa: E402


def prepare_batch(seq_tokens, contact, device):
    import torch
    x = torch.tensor(seq_tokens, dtype=torch.float32, device=device).unsqueeze(0)
    y = torch.tensor(contact, dtype=torch.float32, device=device).unsqueeze(0)
    return x, y
from hgt_psd.model import StructuredPSDModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify HGT-PSD audit ledger")
    parser.add_argument("--train-results", default=str(ROOT / "raw_outputs" / "train_results.json"))
    parser.add_argument("--output", default=str(ROOT / "raw_outputs" / "audit_ledger.json"))
    args = parser.parse_args()

    manifest = load_manifest(ROOT / "manifests" / "gm12878_hic.json")
    fetch_meta = json.loads((ROOT / "raw_outputs" / "fetch_manifest.json").read_text())
    tokens = np.load(ROOT / fetch_meta["tokens_path"])
    contact = np.load(ROOT / fetch_meta["matrix_path"])

    empirical = {
        "data_source": fetch_meta.get("dataset", "ENCODE GM12878 Hi-C"),
        "fetch_mode": fetch_meta.get("fetch_mode", "unknown"),
        "sequence_source": fetch_meta.get("sequence_source"),
        "contact_source": fetch_meta.get("contact_source"),
        "n_bins": fetch_meta.get("n_bins"),
        "demo_mode": fetch_meta.get("demo_mode", False),
    }

    device = torch.device("cpu")
    x, y = prepare_batch(tokens, contact, device)

    train_path = Path(args.train_results)
    train_results = json.loads(train_path.read_text()) if train_path.exists() else {"runs": {}}

    # Proposition 1 check on fresh forward pass
    model = StructuredPSDModel(K=manifest["model_defaults"]["K"], n=manifest["model_defaults"]["n_bins"])
    ckpt = ROOT / "checkpoints" / f"HGT_PSD_K{manifest['model_defaults']['K']}_fourier_stratified.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

    Y_hat = model(x)
    evals = torch.linalg.eigvalsh(Y_hat).detach().numpy()
    target_evals = np.linalg.eigvalsh(contact)

    audit = {
        "empirical_data": empirical,
        "proposition1_psd_valid": model.is_psd(Y_hat),
        "predicted_min_eigenvalue": float(evals.min()),
        "target_min_eigenvalue": float(target_evals.min()),
        "symmetry_error": float(torch.norm(Y_hat - Y_hat.transpose(-1, -2)).item()),
        "train_results_summary": {
            k: {
                "frobenius_rel_error": v.get("frobenius_rel_error"),
                "psd_valid": v.get("psd_valid"),
                "params": v.get("params"),
            }
            for k, v in train_results.get("runs", {}).items()
        },
        "memory_complexity": f"O(n*K + n*d_model), n={manifest['model_defaults']['n_bins']}, K={manifest['model_defaults']['K']}",
    }

    out = Path(args.output)
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("=== HGT-PSD Audit Ledger ===")
    print(f"Data: {empirical['data_source']} ({empirical['fetch_mode']})")
    print(f"Sequence: {empirical.get('sequence_source', 'n/a')}")
    print(f"Proposition 1 (PSD valid): {audit['proposition1_psd_valid']}")
    print(f"Predicted min eigenvalue: {audit['predicted_min_eigenvalue']:.6f}")
    print(f"Symmetry error: {audit['symmetry_error']:.2e}")
    for name, m in audit["train_results_summary"].items():
        print(f"  {name}: rel_err={m['frobenius_rel_error']:.4f} psd={m['psd_valid']}")
    print(f"\nAudit -> {out}")


if __name__ == "__main__":
    main()
