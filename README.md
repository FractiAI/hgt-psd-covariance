# HGT-PSD Covariance — Reproducible Chromatin Contact Modeling

**Hierarchical Genomic Tokenization + Structured PSD Covariance Operators** for Hi-C contact map prediction with PSD-by-construction guarantees.

**GitHub:** [github.com/FractiAI/hgt-psd-covariance](https://github.com/FractiAI/hgt-psd-covariance)  
**Paper:** `paper/HGT_PSD_COVARIANCE.md`  
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

### Windows

```powershell
.\verify_pipeline.ps1
```

### Linux / macOS

```bash
chmod +x setup_env.sh verify_pipeline.sh
./verify_pipeline.sh
```

---

## Full pipeline

```bash
python tools/fetch_gm12878_hic.py          # or --demo for synthetic PSD matrix
python tools/train.py --ablation-rank
python tools/verify_audit.py
```

Outputs: `raw_outputs/train_results.json`, `raw_outputs/audit_ledger.json`

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
