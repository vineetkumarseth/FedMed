#!/usr/bin/env python3
"""Privacy/utility trade-off: sweep the DP noise multiplier and report (epsilon, Dice).

Runs FedAvg *in-process* (no gRPC, no encryption - those do not change the maths, since
CKKS aggregation is exact up to ~1e-7) so a sweep takes minutes instead of hours.

    python scripts/privacy_sweep.py --config configs/default.yaml --sigmas 0 0.002 0.01 0.05 0.2
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from fedmed.config import load_config
from fedmed.data import load_hospital_data
from fedmed.models import build_model
from fedmed.privacy.dp import RDPAccountant, add_gaussian_noise, clip_update
from fedmed.training.local import evaluate, get_flat, resolve_device, set_flat, train_epochs


def simulate(cfg, sigma: float, rounds: int):
    device = resolve_device(cfg.train.device)
    torch.manual_seed(cfg.train.seed)
    hospitals = {h: load_hospital_data(cfg, h) for h in range(1, cfg.federation.num_hospitals + 1)}
    model = build_model(cfg)
    global_flat = get_flat(model)
    rng = np.random.default_rng(0)
    for r in range(1, rounds + 1):
        acc, total = np.zeros_like(global_flat, dtype=np.float64), 0
        for h, (train, _) in hospitals.items():
            set_flat(model, global_flat)
            train_epochs(model, train, epochs=cfg.train.local_epochs, batch_size=cfg.train.batch_size,
                         lr=cfg.train.lr, device=device, seed=cfg.train.seed + 97 * r + h)
            delta = get_flat(model) - global_flat
            if sigma >= 0:
                delta, _, _ = clip_update(delta, cfg.dp.clip_norm)
                delta = add_gaussian_noise(delta, cfg.dp.clip_norm, sigma, rng)
            acc += len(train) * (global_flat + delta)
            total += len(train)
        global_flat = (acc / total).astype(np.float32)
    set_flat(model, global_flat)
    dices = [evaluate(model, val, batch_size=cfg.train.batch_size, device=device) for _, val in hospitals.values()]
    n = sum(d["n"] for d in dices)
    return sum(d["dice"] * d["n"] for d in dices) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[-1, 0.002, 0.01, 0.05, 0.2],
                    help="noise multipliers; -1 = no clipping and no noise")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--out", default="runs/privacy_sweep.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rounds = args.rounds or cfg.federation.rounds
    rows = []
    print(f"{'sigma':>8} {'epsilon':>12} {'Dice':>7}   (delta={cfg.dp.delta}, {rounds} rounds, clip={cfg.dp.clip_norm})")
    for s in args.sigmas:
        dice = simulate(cfg, s, rounds)
        eps = None if s < 0 else (RDPAccountant(s, cfg.dp.delta).epsilon(rounds) if s > 0 else math.inf)
        rows.append({"sigma": s, "epsilon": None if eps is None or math.isinf(eps) else eps, "dice": dice})
        label = "no DP" if s < 0 else f"{eps:12.3g}"
        print(f"{s:8.3g} {label:>12} {dice:7.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rounds": rounds, "clip_norm": cfg.dp.clip_norm, "rows": rows}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
