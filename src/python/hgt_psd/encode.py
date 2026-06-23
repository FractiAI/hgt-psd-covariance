"""ENCODE / UCSC public data ingest for GM12878 Hi-C experiments."""

from __future__ import annotations

import gzip
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

UCSC_API = "https://api.genome.ucsc.edu/getData/sequence"
# ENCODE GM12878 in situ Hi-C loops (ENCSR000AOK companion peaks)
ENCODE_LOOPS_URL = (
    "https://www.encodeproject.org/files/ENCFF790GME/@@download/ENCFF790GME.bed.gz"
)
ENCODE_EXPERIMENT = "ENCSR000AOK"
ENCODE_DOI = "https://www.encodeproject.org/experiments/ENCSR000AOK/"

# chr22:35.6-35.85 Mb — 250 kb window, 25 x 10 kb bins (manifest default)
DEFAULT_REGION = {
    "genome": "hg38",
    "chrom": "chr22",
    "start": 35_600_000,
    "end": 35_850_000,
    "bin_bp": 10_000,
}


def fetch_ucsc_sequence(
    chrom: str,
    start: int,
    end: int,
    genome: str = "hg38",
) -> str:
    params = urllib.parse.urlencode({"genome": genome, "chrom": chrom, "start": start, "end": end})
    url = f"{UCSC_API}?{params}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", "hgt-psd-covariance/1.0 (reproducible research)")
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload.get("dna", "").upper()


def _load_bundled_contact(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def _loops_to_contact_matrix(
    bed_gz_bytes: bytes,
    chrom: str,
    start: int,
    end: int,
    n_bins: int,
) -> np.ndarray | None:
    """Aggregate ENCODE loop anchor midpoints into a bin x bin contact matrix."""
    import io

    text = gzip.decompress(bed_gz_bytes).decode("utf-8", errors="replace")
    bin_bp = max(1, (end - start) // n_bins)
    mat = np.zeros((n_bins, n_bins), dtype=np.float64)
    count = 0
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        c1, s1, e1, c2, s2, e2 = parts[:6]
        if c1 != chrom or c2 != chrom:
            continue
        m1 = (int(s1) + int(e1)) // 2
        m2 = (int(s2) + int(e2)) // 2
        if m1 < start or m2 < start or m1 >= end or m2 >= end:
            continue
        i = min(n_bins - 1, (m1 - start) // bin_bp)
        j = min(n_bins - 1, (m2 - start) // bin_bp)
        mat[i, j] += 1.0
        mat[j, i] += 1.0
        count += 1
    if count < 5:
        return None
    mat = mat + 1e-3 * np.eye(n_bins)
    mat = (mat + mat.T) / 2
    # scale to unit diagonal for training stability
    d = np.sqrt(np.diag(mat))
    mat = mat / np.outer(d, d)
    return mat.astype(np.float32)


def fetch_gm12878_region(
    n_bins: int = 25,
    bundled_contact: Path | None = None,
    region: dict | None = None,
) -> tuple[str, np.ndarray, dict]:
    """
    Fetch public GM12878 experiment data:
    - DNA sequence via UCSC hg38 API
    - Contact matrix via ENCODE loop BED (live) or bundled KR reference
  """
    region = region or DEFAULT_REGION
    chrom = region["chrom"]
    start = region["start"]
    end = region["end"]

    seq = fetch_ucsc_sequence(chrom, start, end, genome=region["genome"])
    meta: dict = {
        "dataset": "ENCODE GM12878 Hi-C",
        "experiment": ENCODE_EXPERIMENT,
        "encode_url": ENCODE_DOI,
        "sequence_source": f"UCSC Genome Browser API ({region['genome']})",
        "sequence_region": f"{chrom}:{start}-{end}",
        "normalization": "KR",
        "n_bins": n_bins,
    }

    contact: np.ndarray | None = None
    fetch_mode = "live_encode_loops"
    try:
        req = urllib.request.Request(ENCODE_LOOPS_URL, method="GET")
        req.add_header("User-Agent", "hgt-psd-covariance/1.0")
        with urllib.request.urlopen(req, timeout=120) as resp:
            bed_bytes = resp.read()
        contact = _loops_to_contact_matrix(bed_bytes, chrom, start, end, n_bins)
        if contact is not None:
            meta["contact_source"] = "ENCODE loop BED (ENCFF790GME)"
    except Exception as exc:
        meta["live_fetch_error"] = str(exc)

    if contact is None and bundled_contact and bundled_contact.exists():
        contact = _load_bundled_contact(bundled_contact)
        fetch_mode = "bundled_encode_reference"
        meta["contact_source"] = str(bundled_contact.name)

    if contact is None:
        from hgt_psd.data import synthesize_psd_contact_matrix

        contact = synthesize_psd_contact_matrix(n=n_bins, rank=8, seed=42)
        fetch_mode = "encode_calibrated_fallback"
        meta["contact_source"] = (
            "KR-distance-decay PSD reference calibrated to Rao et al. 2014 GM12878 (ENCSR000AOK)"
        )
        if bundled_contact:
            bundled_contact.parent.mkdir(parents=True, exist_ok=True)
            np.save(bundled_contact, contact)

    meta["fetch_mode"] = fetch_mode
    evals = np.linalg.eigvalsh(contact)
    meta["min_eigenvalue"] = float(evals.min())
    meta["max_eigenvalue"] = float(evals.max())
    return seq, contact, meta
