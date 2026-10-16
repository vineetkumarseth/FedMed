"""Centralized baseline (Week 1): train one U-Net on the *pooled* data.

This deliberately violates the privacy boundary - it exists only to measure the accuracy
ceiling the federation is trying to approach. Never do this with real patient data.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .config import Config
from .data import load_centralized
from .models import build_model
from .models.factory import count_parameters
from .training.local import evaluate, resolve_device, train_epochs


def run_baseline(cfg: Config, run_dir: Path, epochs: int | None = None) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.train.device)
    torch.manual_seed(cfg.train.seed)

    train_ds, val_ds = load_centralized(cfg)
    model = build_model(cfg)
    # Same number of passes over every scan as the federation makes (rounds x local epochs).
    epochs = epochs or cfg.federation.rounds * cfg.train.local_epochs
    curve, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        tr = train_epochs(model, train_ds, epochs=1, batch_size=cfg.train.batch_size, lr=cfg.train.lr,
                          device=device, seed=cfg.train.seed + ep)
        va = evaluate(model, val_ds, batch_size=cfg.train.batch_size, device=device)
        curve.append({"epoch": ep, "train_loss": tr["loss"], "val_loss": va["loss"], "val_dice": va["dice"]})
        print(f"[baseline] epoch {ep:>3}/{epochs}  loss {tr['loss']:.4f}  val dice {va['dice']:.4f}")
    result = {"epochs": epochs, "final_dice": curve[-1]["val_dice"], "best_dice": max(c["val_dice"] for c in curve),
              "n_params": count_parameters(model), "wall_seconds": round(time.time() - t0, 1), "curve": curve}
    (run_dir / "baseline.json").write_text(json.dumps(result, indent=2))
    torch.save({"state_dict": model.state_dict()}, run_dir / "baseline_model.pt")
    return result
