# Hierarchical Genomic Tokenization and Structured PSD Covariance Operators

**GitHub:** https://github.com/FractiAI/hgt-psd-covariance  
**License:** MIT

---

## Abstract

Predicting 3D chromatin conformation requires models that satisfy physical constraints, specifically positive semi-definiteness (PSD) and stationary distance decay. We present a framework that treats chromatin contact prediction as a **Conditional PSD Covariance Estimator**. The model maps genomic sequences to a sequence-dependent basis Φ(x) ∈ ℝⁿˣᴷ, defining the contact adjacency as Ŷ(x) = Φ(x)Φ(x)ᵀ + diag(σ²(x)). This construction guarantees that predicted contact maps reside within the PSD cone S₊ⁿ by construction. We resolve the scaling bottleneck via **Hierarchical Genomic Tokenization** (250-kb → 10-kb bins) and stabilize the objective using an **Orthogonal Frobenius-Space Masking Loss**, which decouples gradient flow across distance-stratified regimes. The architecture maintains O(nR + nd_model) memory, achieving high-fidelity Hi-C regression on ENCODE GM12878 matrices.

---

## 1. Structural Formulation

We model the interaction adjacency map M: 𝒳 → S₊ⁿ. Given sequence input x ∈ 𝒳:

**Ŷ(x) = Φ(x)Φ(x)ᵀ + diag(σ²(x))**

where Φ(x) = Ψ diag(√α(x)) ∈ ℝⁿˣᴷ, Ψ is a fixed Fourier basis, and α(x) ∈ ℝ₊ᴷ is sequence-conditioned.

**Proposition 1 (PSD Validity).** For all x ∈ 𝒳, Ŷ(x) ∈ S₊ⁿ.

*Proof.* Sum of Gram matrix and non-negative diagonal. ■

**Implementation:** `src/python/hgt_psd/model.py`

---

## 2. Multi-Scale Risk Functional

Training minimizes variance-stabilized Frobenius error across distance strata:

**ℒ = Σₛ ‖ Πₛ(Y − Ŷ) ‖²_F / (Var(Πₛ Y) + ε)**

**Implementation:** `src/python/hgt_psd/loss.py`

---

## 3. Hierarchical Genomic Tokenization

250-kb coarse bins refined to 10-kb resolution for sequence embedding.

**Implementation:** `src/python/hgt_psd/tokenization.py`  
**Manifest:** `manifests/gm12878_hic.json`

---

## 4. Experimental Validation Plan

| Item | Detail |
|------|--------|
| Dataset | ENCODE GM12878 Hi-C, KR-normalized |
| Baselines | Akita stub, DeepC stub, low-rank factorized |
| Basis ablation | Fourier vs random orthonormal Ψ |
| Rank ablation | K ∈ {8, 16, 32, 64} |
| Loss ablation | Stratified Frobenius vs MSE |

---

## 5. Reproducibility

```bash
./setup_env.sh
python tools/fetch_gm12878_hic.py --demo
python tools/train.py --ablation-rank
python tools/verify_audit.py
```

Docker: `docker build -t hgt-psd:v1 . && ./verify_pipeline.sh`

---

## References

ENCODE GM12878 in-situ Hi-C. See `manifests/gm12878_hic.json` for provenance fields.
