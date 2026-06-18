"""Structured low-rank PSD covariance model for Hi-C contact prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredPSDModel(nn.Module):
    """
    Conditional PSD Covariance Estimator.

    Y_hat(x) = Phi(x) Phi(x)^T + diag(sigma^2(x))
    Phi(x) = Psi @ diag(sqrt(alpha(x)))
    """

    def __init__(
        self,
        d_model: int = 512,
        K: int = 32,
        n: int = 25,
        basis: str = "fourier",
        seed: int = 42,
    ):
        super().__init__()
        self.n = n
        self.K = K
        self.d_model = d_model
        self.basis_type = basis

        self.alpha_map = nn.Linear(d_model, K)
        self.sigma_map = nn.Linear(d_model, n)
        self.input_proj = nn.Linear(4, d_model)  # ACGT one-hot bins

        Psi = self._build_basis(n, K, basis, seed)
        self.register_buffer("Psi", Psi)

    @staticmethod
    def _build_basis(n: int, K: int, basis: str, seed: int) -> torch.Tensor:
        if basis == "fourier":
            freqs = torch.linspace(0.1, 2.0, K)
            i = torch.arange(n, dtype=torch.float32)
            return torch.cos(i.unsqueeze(-1) * freqs.unsqueeze(0))
        # Random orthonormal basis (ablation)
        g = torch.Generator().manual_seed(seed)
        M = torch.randn(n, K, generator=g)
        q, _ = torch.linalg.qr(M)
        return q[:, :K]

    def encode_sequence(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        """seq_tokens: (batch, tokens, 4) one-hot -> (batch, d_model) pooled embedding."""
        h = self.input_proj(seq_tokens)
        return h.mean(dim=1)

    def forward(
        self, seq_tokens: torch.Tensor, return_components: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        seq_tokens: (batch, tokens, 4)
        returns Y_hat: (batch, n, n) PSD by construction
        """
        x = self.encode_sequence(seq_tokens)
        alpha = F.softmax(self.alpha_map(x), dim=-1)
        sigma2 = F.softplus(self.sigma_map(x)) + 1e-8

        # Phi: (batch, n, K) = Psi(n,K) * sqrt(alpha)(batch,K)
        sqrt_alpha = torch.sqrt(alpha + 1e-8)
        Phi = self.Psi.unsqueeze(0) * sqrt_alpha.unsqueeze(1)

        # Gram + diagonal noise
        Y = torch.einsum("bnk,bmk->bnm", Phi, Phi)
        Y = Y + torch.diag_embed(sigma2)

        if return_components:
            return Y, {"alpha": alpha, "sigma2": sigma2, "Phi": Phi}
        return Y

    def is_psd(self, Y: torch.Tensor, tol: float = -1e-5) -> bool:
        """Check minimum eigenvalue >= tol for batch of matrices."""
        evals = torch.linalg.eigvalsh(Y)
        return bool((evals.min() >= tol).item())

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
