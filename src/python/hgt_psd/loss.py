"""Orthogonal Frobenius-space masking loss (variance-stabilized strata)."""

from __future__ import annotations

import torch
import torch.nn as nn


def distance_strata_masks(n: int, strata: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Build boolean masks Pi_s for each distance stratum."""
    idx = torch.arange(n, device=device)
    dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
    masks = {}
    for name, bounds in strata.items():
        lo = bounds.get("min_dist", 0)
        hi = bounds.get("max_dist")
        if hi is None:
            m = dist >= lo
        else:
            m = (dist >= lo) & (dist <= hi)
        masks[name] = m.float()
    return masks


class StratifiedFrobeniusLoss(nn.Module):
    """
    L = sum_s || Pi_s (Y - Y_hat) ||_F^2 / (Var(Pi_s Y) + eps)
    """

    def __init__(self, n: int, strata: dict, epsilon: float = 1e-6):
        super().__init__()
        self.n = n
        self.strata = strata
        self.epsilon = epsilon
        self._masks: dict[str, torch.Tensor] | None = None

    def _get_masks(self, device: torch.device) -> dict[str, torch.Tensor]:
        if self._masks is None or next(iter(self._masks.values())).device != device:
            self._masks = distance_strata_masks(self.n, self.strata, device)
        return self._masks

    def forward(
        self, Y_hat: torch.Tensor, Y: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        masks = self._get_masks(Y.device)
        total = torch.tensor(0.0, device=Y.device)
        terms: dict[str, float] = {}
        for name, Pi in masks.items():
            diff = (Y - Y_hat) * Pi
            target = Y * Pi
            num = (diff ** 2).sum(dim=(-2, -1))
            denom = target.var(dim=(-2, -1), unbiased=False) + self.epsilon
            term = (num / denom).mean()
            total = total + term
            terms[name] = float(term.item())
        return total, terms


class MSELoss(nn.Module):
    """Standard MSE ablation baseline."""

    def forward(self, Y_hat: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        loss = ((Y - Y_hat) ** 2).mean()
        return loss, {"mse": float(loss.item())}
