#!/usr/bin/env python3
"""Fetch ENCODE GM12878 Hi-C region (UCSC sequence + contact matrix)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))

from hgt_psd.data import load_manifest, save_matrix  # noqa: E402
from hgt_psd.encode import fetch_gm12878_region  # noqa: E402
from hgt_psd.tokenization import HierarchicalGenomicTokenizer  # noqa: E402

OUT_DIR = ROOT / "data" / "matrices"
META_OUT = ROOT / "raw_outputs" / "fetch_manifest.json"
BUNDLED_CONTACT = ROOT / "data" / "reference" / "gm12878_chr22_kr_25x25.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GM12878 Hi-C data")
    parser.add_argument("--manifest", default=str(ROOT / "manifests" / "gm12878_hic.json"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="Synthetic smoke test only")
    mode.add_argument("--public", action="store_true", help="ENCODE/UCSC public ingest (default)")
    parser.add_argument("--n-bins", type=int, default=25)
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    n = args.n_bins
    seed = manifest["model_defaults"].get("seed", 42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = OUT_DIR / "gm12878_contact.npy"
    seq_path = OUT_DIR / "gm12878_sequence.txt"
    tokens_path = OUT_DIR / "gm12878_tokens.npy"

    if args.demo:
        from hgt_psd.data import synthesize_psd_contact_matrix, synthesize_sequence

        contact = synthesize_psd_contact_matrix(n=n, rank=8, seed=seed)
        seq_len = manifest["hierarchical_tokenization"]["coarse_bin_bp"] // 1000
        seq = synthesize_sequence(seq_len * 1000, seed=seed)
        meta = {
            "dataset": manifest["dataset"],
            "fetch_mode": "demo_synthetic",
            "demo_mode": True,
            "sequence_source": "synthetic",
            "contact_source": "synthetic PSD",
        }
    else:
        seq, contact, meta = fetch_gm12878_region(
            n_bins=n,
            bundled_contact=BUNDLED_CONTACT,
        )
        meta["demo_mode"] = False

    save_matrix(matrix_path, contact)
    seq_path.write_text(seq[: manifest["hierarchical_tokenization"]["coarse_bin_bp"]], encoding="utf-8")

    tok = HierarchicalGenomicTokenizer(
        coarse_bin_bp=manifest["hierarchical_tokenization"]["coarse_bin_bp"],
        fine_bin_bp=manifest["hierarchical_tokenization"]["fine_bin_bp"],
    )
    tokens = tok.tokenize_region(seq_path.read_text(), n)
    np.save(tokens_path, tokens)

    evals = np.linalg.eigvalsh(contact)
    meta.update({
        "normalization": manifest["normalization"],
        "n_bins": n,
        "matrix_path": str(matrix_path.relative_to(ROOT)),
        "sequence_path": str(seq_path.relative_to(ROOT)),
        "tokens_path": str(tokens_path.relative_to(ROOT)),
        "target_psd": True,
        "min_eigenvalue": float(evals.min()),
        "max_eigenvalue": float(evals.max()),
    })
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Mode: {meta.get('fetch_mode', 'public')}")
    print(f"Sequence: {meta.get('sequence_source', 'n/a')}")
    print(f"Contact: {meta.get('contact_source', 'n/a')}")
    print(f"Contact matrix {contact.shape} saved -> {matrix_path}")
    print(f"PSD check: min eigenvalue = {meta['min_eigenvalue']:.6f}")
    print(f"Manifest -> {META_OUT}")


if __name__ == "__main__":
    main()
