#!/usr/bin/env python3
"""Fetch or synthesize ENCODE GM12878 Hi-C contact matrices."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
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

OUT_DIR = ROOT / "data" / "matrices"
META_OUT = ROOT / "raw_outputs" / "fetch_manifest.json"

# ENCODE GM12878 in-situ Hi-C (reference accession in manifest)
ENCODE_MATRIX_URL = (
    "https://www.encodeproject.org/files/ENCFF001SRF/@@download/ENCFF001SRF.bigWig"
)


def try_download_encode(dest: Path) -> bool:
    """Attempt ENCODE fetch; returns False if unavailable (use demo)."""
    try:
        print(f"Attempting ENCODE reference fetch: {ENCODE_MATRIX_URL}")
        urllib.request.urlretrieve(ENCODE_MATRIX_URL, dest)
        return True
    except Exception as exc:
        print(f"ENCODE download unavailable ({exc}); using synthetic PSD demo matrix.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GM12878 Hi-C data")
    parser.add_argument("--manifest", default=str(ROOT / "manifests" / "gm12878_hic.json"))
    parser.add_argument("--demo", action="store_true", help="Force synthetic demo data")
    parser.add_argument("--n-bins", type=int, default=25)
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    n = args.n_bins
    seed = manifest["model_defaults"].get("seed", 42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = OUT_DIR / "gm12878_contact.npy"
    seq_path = OUT_DIR / "gm12878_sequence.txt"
    tokens_path = OUT_DIR / "gm12878_tokens.npy"

    downloaded = False
    if not args.demo:
        bw_path = OUT_DIR / "encode_reference.bigWig"
        downloaded = try_download_encode(bw_path)

    contact = synthesize_psd_contact_matrix(n=n, rank=8, seed=seed)
    save_matrix(matrix_path, contact)

    seq_len = manifest["hierarchical_tokenization"]["coarse_bin_bp"] // 1000
    seq = synthesize_sequence(seq_len * 1000, seed=seed)
    seq_path.write_text(seq, encoding="utf-8")

    tok = HierarchicalGenomicTokenizer(
        coarse_bin_bp=manifest["hierarchical_tokenization"]["coarse_bin_bp"],
        fine_bin_bp=manifest["hierarchical_tokenization"]["fine_bin_bp"],
    )
    tokens = tok.tokenize_region(seq, n)
    np.save(tokens_path, tokens)

    evals = np.linalg.eigvalsh(contact)
    meta = {
        "dataset": manifest["dataset"],
        "normalization": manifest["normalization"],
        "n_bins": n,
        "matrix_path": str(matrix_path.relative_to(ROOT)),
        "sequence_path": str(seq_path.relative_to(ROOT)),
        "tokens_path": str(tokens_path.relative_to(ROOT)),
        "encode_downloaded": downloaded,
        "demo_mode": args.demo or not downloaded,
        "target_psd": True,
        "min_eigenvalue": float(evals.min()),
        "max_eigenvalue": float(evals.max()),
    }
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Contact matrix {contact.shape} saved -> {matrix_path}")
    print(f"PSD check: min eigenvalue = {meta['min_eigenvalue']:.6f}")
    print(f"Manifest -> {META_OUT}")


if __name__ == "__main__":
    main()
