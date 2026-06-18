# HGT-PSD Covariance — Reproducible Chromatin Contact Modeling

## Intention

Chromatin contact maps (Hi-C) describe which parts of the genome are physically close inside the nucleus. Many prediction models treat these maps as arbitrary matrices. In reality, a valid contact map must obey physical constraints: it must be **symmetric**, **positive semi-definite (PSD)**, and show **distance decay** (nearby loci interact more strongly).

This repository exists to make that constraint **built in, not bolted on**. We provide a reproducible pipeline that:

1. Predicts Hi-C contact maps from DNA sequence using a **structured PSD covariance operator** — every prediction is PSD-valid by construction.
2. Scales to realistic genomic windows via **hierarchical tokenization** (250-kb regions refined to 10-kb bins).
3. Trains with a **distance-stratified loss** so gradients are balanced across short-, mid-, and long-range contacts.
4. Ships with locked manifests, audit scripts, and ablations so results can be verified independently.

If you want to predict chromatin contacts while respecting physical structure — or compare against baselines that do not — start here.

---

## Abstract

We frame chromatin contact prediction as estimating a **conditional PSD covariance matrix** from sequence. A genomic sequence is embedded into a low-rank basis Φ(x); the predicted contact map is:

**Ŷ(x) = Φ(x)Φ(x)ᵀ + diag(σ²(x))**

Because Ŷ is a Gram matrix plus a non-negative diagonal, it lies in the PSD cone **by construction** (Proposition 1 in the paper). Hierarchical genomic tokenization (250-kb → 10-kb) keeps memory at **O(nK + nd_model)** instead of materializing full n×n dense operators naively. Training uses an orthogonal Frobenius-space masking loss that stratifies error by genomic distance.

**Key findings from this implementation:**

| Finding | What it means |
|---------|---------------|
| **PSD validity is guaranteed** | Every forward pass produces a symmetric, PSD contact map — verified automatically by `verify_audit.py` (minimum eigenvalue ≥ 0). |
| **Structured beats unconstrained factorization** | Low-rank models without PSD structure can violate physical constraints; our parameterization cannot. |
| **Hierarchical tokenization enables scale** | Coarse 250-kb bins anchor context; 10-kb bins carry sequence detail without exploding memory. |
| **Stratified loss stabilizes training** | Distance-stratified Frobenius masking prevents long-range noise from dominating gradients. |
| **Reproducible end-to-end** | ENCODE GM12878 ingest, training, rank/basis/loss ablations, and an audit ledger are scripted and manifest-locked. |

Full mathematical treatment: [`paper/HGT_PSD_COVARIANCE.md`](paper/HGT_PSD_COVARIANCE.md)

---

## Primer — concepts before you run anything

**Hi-C contact map**  
A square matrix where entry (i, j) measures how often genomic bin *i* contacts bin *j*. Rows and columns index bins along a chromosome region. Strong values on the diagonal and nearby off-diagonal entries reflect 3D proximity.

**Why PSD matters**  
A PSD matrix has no negative eigenvalues. Contact maps derived from true 3D structures satisfy this. Models that output arbitrary matrices can predict physically impossible interactions. We avoid that by architecture, not post-hoc projection.

**Hierarchical genomic tokenization (HGT)**  
Instead of one token per base pair, we use two scales: **250-kb coarse bins** for regional context and **10-kb fine bins** for local sequence. This matches how Hi-C data is binned and keeps compute tractable.

**Structured PSD covariance operator**  
The model does not learn a raw n×n matrix. It learns (a) sequence-dependent weights on a fixed basis Ψ and (b) per-bin variance terms σ². The product ΦΦᵀ is always PSD.

**Stratified Frobenius loss**  
Contacts at different genomic distances have different variance. We mask the error matrix into distance bands and normalize each band by its empirical variance, so no single distance regime swamps training.

**What you get when you run the pipeline**  
- `raw_outputs/train_results.json` — training metrics and ablation runs  
- `raw_outputs/audit_ledger.json` — PSD validity check, eigenvalue bounds, symmetry error  
- `checkpoints/` — saved model weights (after training)

---

## Links

**GitHub:** [github.com/FractiAI/hgt-psd-covariance](https://github.com/FractiAI/hgt-psd-covariance)  
**Paper:** [`paper/HGT_PSD_COVARIANCE.md`](paper/HGT_PSD_COVARIANCE.md)  
**License:** MIT

---

## What this repo contains

- **StructuredPSDModel** — Ŷ = ΦΦᵀ + diag(σ²), PSD valid by construction (Proposition 1)
- **Stratified Frobenius loss** — distance-stratified orthogonal masking
- **Hierarchical tokenization** — 250-kb → 10-kb genomic bins
- **ENCODE GM12878 pipeline** — fetch, train, verify audit ledger
- **Baselines** — Akita/DeepC stubs + unconstrained low-rank factorization
- **Rank/basis/loss ablations** — `--ablation-rank`, `--basis`, `--loss`

---

## Quick start

The fastest way to confirm everything works on your machine:

### Windows

```powershell
.\verify_pipeline.ps1
```

> If PyTorch cannot load (common without the [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)), the script falls back to a NumPy smoke test that still verifies PSD validity and core math. See [`VALIDATION.md`](VALIDATION.md).

### Linux / macOS

```bash
chmod +x setup_env.sh verify_pipeline.sh
./verify_pipeline.sh
```

### Docker

```bash
docker build -t hgt-psd:v1 .
docker run --rm -v "$(pwd)":/workspace hgt-psd:v1
```

---

## Full pipeline

Run each step individually when you want control over data source, epochs, or ablations:

```bash
python tools/fetch_gm12878_hic.py          # ENCODE ingest; add --demo for synthetic PSD matrix
python tools/train.py --ablation-rank      # train + rank ablation (K ∈ {8, 16, 32, 64})
python tools/verify_audit.py               # PSD audit ledger
```

**Outputs:** `raw_outputs/train_results.json`, `raw_outputs/audit_ledger.json`

**Useful flags:**

| Flag | Effect |
|------|--------|
| `--demo` | Synthetic PSD contact matrix (no network fetch) |
| `--ablation-rank` | Train across multiple rank values K |
| `--basis random` | Ablation: random orthonormal Ψ instead of Fourier |
| `--loss mse` | Ablation: plain MSE instead of stratified Frobenius |
| `--epochs N` | Training epochs (default in manifest) |

Dataset and hyperparameters are locked in [`manifests/gm12878_hic.json`](manifests/gm12878_hic.json).

---

## Repository layout

| Path | Purpose |
|------|---------|
| `paper/HGT_PSD_COVARIANCE.md` | Manuscript |
| `manifests/gm12878_hic.json` | Dataset + hyperparameter lock |
| `src/python/hgt_psd/model.py` | StructuredPSDModel |
| `src/python/hgt_psd/loss.py` | Stratified Frobenius masking loss |
| `src/python/hgt_psd/tokenization.py` | Hierarchical genomic tokenization |
| `tools/fetch_gm12878_hic.py` | ENCODE ingest / demo synthesis |
| `tools/train.py` | Training + ablations |
| `tools/verify_audit.py` | PSD validity audit |
| `VALIDATION.md` | Platform notes and smoke-test details |

---

## Citation

```bibtex
@article{hgt_psd_covariance_2026,
  title={Hierarchical Genomic Tokenization and Structured PSD Covariance Operators},
  author={FractiAI},
  year={2026},
  note={https://github.com/FractiAI/hgt-psd-covariance}
}
```
