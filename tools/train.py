#!/usr/bin/env python3
"""Train StructuredPSDModel on GM12878 Hi-C contact matrix."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))

from hgt_psd.baselines import AkitaStub, DeepCStub, LowRankFactorized  # noqa: E402
from hgt_psd.data import load_manifest  # noqa: E402
from hgt_psd.loss import MSELoss, StratifiedFrobeniusLoss  # noqa: E402
from hgt_psd.model import StructuredPSDModel  # noqa: E402


def prepare_batch(seq_tokens, contact, device):
    x = torch.tensor(seq_tokens, dtype=torch.float32, device=device).unsqueeze(0)
    y = torch.tensor(contact, dtype=torch.float32, device=device).unsqueeze(0)
    return x, y


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def frobenius_error(Y_hat: torch.Tensor, Y: torch.Tensor) -> float:
    return float(torch.norm(Y - Y_hat, p="fro").item() / torch.norm(Y, p="fro").item())


def train_model(
    model: torch.nn.Module,
    loss_fn,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        Y_hat = model(x)
        loss, terms = loss_fn(Y_hat, y)
        loss.backward()
        opt.step()
        history.append({"epoch": ep + 1, "loss": float(loss.item()), **terms})
    return {
        "final_loss": history[-1]["loss"],
        "frobenius_rel_error": frobenius_error(model(x).detach(), y),
        "wall_seconds": time.perf_counter() - t0,
        "epochs": epochs,
        "params": sum(p.numel() for p in model.parameters()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PSD chromatin model")
    parser.add_argument("--manifest", default=str(ROOT / "manifests" / "gm12878_hic.json"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--basis", default="fourier", choices=["fourier", "random"])
    parser.add_argument("--loss", default="stratified", choices=["stratified", "mse"])
    parser.add_argument("--ablation-rank", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "raw_outputs" / "train_results.json"))
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    defaults = manifest["model_defaults"]
    seed = defaults["seed"]
    n = defaults["n_bins"]
    K = args.K or defaults["K"]
    set_seed(seed)

    fetch_meta = json.loads((ROOT / "raw_outputs" / "fetch_manifest.json").read_text())
    tokens = np.load(ROOT / fetch_meta["tokens_path"])
    contact = np.load(ROOT / fetch_meta["matrix_path"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = prepare_batch(tokens, contact, device)

    strata = manifest["loss_strata"]
    epsilon = manifest.get("epsilon", 1e-6)

    results: dict = {"seed": seed, "device": str(device), "runs": {}}

    ranks = defaults["rank_ablation"] if args.ablation_rank else [K]

    for rank in ranks:
        model = StructuredPSDModel(
            d_model=defaults["d_model"],
            K=rank,
            n=n,
            basis=args.basis,
            seed=seed,
        ).to(device)

        if args.loss == "stratified":
            loss_fn = StratifiedFrobeniusLoss(n=n, strata=strata, epsilon=epsilon)
        else:
            loss_fn = MSELoss()

        run_id = f"HGT_PSD_K{rank}_{args.basis}_{args.loss}"
        print(f"\n=== Training {run_id} ===")
        metrics = train_model(model, loss_fn, x, y, epochs=args.epochs)
        psd_ok = model.is_psd(model(x).detach())
        metrics["psd_valid"] = psd_ok
        metrics["K"] = rank
        metrics["basis"] = args.basis
        metrics["loss"] = args.loss

        ckpt = ROOT / "checkpoints" / f"{run_id}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        metrics["checkpoint"] = str(ckpt.relative_to(ROOT))
        results["runs"][run_id] = metrics
        print(f"  rel Frobenius error: {metrics['frobenius_rel_error']:.4f}")
        print(f"  PSD valid: {psd_ok}")

    # Baseline stubs
    for name, cls in [("LowRankFactorized", LowRankFactorized), ("DeepCStub", DeepCStub), ("AkitaStub", AkitaStub)]:
        print(f"\n=== Baseline {name} ===")
        m = cls(n=n).to(device)
        metrics = train_model(m, MSELoss(), x, y, epochs=args.epochs)
        evals = torch.linalg.eigvalsh(m(x).detach())
        metrics["psd_valid"] = bool((evals.min() >= -1e-4).item())
        metrics["min_eigenvalue"] = float(evals.min().item())
        results["runs"][name] = metrics
        print(f"  rel Frobenius error: {metrics['frobenius_rel_error']:.4f}")
        print(f"  min eigenvalue: {metrics.get('min_eigenvalue', 'n/a')}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
