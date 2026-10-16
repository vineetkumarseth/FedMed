"""Local (on-premise) training and evaluation loops shared by clients and the baseline."""
from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .losses import segmentation_loss
from .metrics import dice_and_iou


def set_threads(n: int) -> None:
    if n and n > 0:
        torch.set_num_threads(n)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def get_flat(model: nn.Module) -> np.ndarray:
    """All parameters as one float32 vector (the model *is* its parameters: no BN buffers)."""
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()]).astype(np.float32)


def set_flat(model: nn.Module, flat: np.ndarray) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(torch.from_numpy(np.asarray(flat[offset:offset + n], dtype=np.float32)).view_as(p))
            offset += n
    if offset != len(flat):
        raise ValueError(f"Parameter vector length mismatch: model={offset}, vector={len(flat)}")


def train_epochs(model: nn.Module, dataset: Dataset, *, epochs: int, batch_size: int, lr: float,
                 device: torch.device, seed: int, proximal_mu: float = 0.0,
                 global_flat: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Run ``epochs`` of local training. Returns the mean loss/dice of the last epoch."""
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    anchor = None
    if proximal_mu > 0 and global_flat is not None:
        anchor = [torch.from_numpy(a).to(device) for a in _split_like(model, global_flat)]

    model.to(device).train()
    t0 = time.perf_counter()
    last = {"loss": 0.0, "dice": 0.0}
    for _ in range(epochs):
        losses, dices = [], []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = segmentation_loss(logits, y)
            if anchor is not None:  # FedProx: keep the local model close to the global one
                prox = sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
                loss = loss + 0.5 * proximal_mu * prox
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            dices.append(float(dice_and_iou(logits.detach(), y)[0].mean()))
        last = {"loss": float(np.mean(losses)), "dice": float(np.mean(dices))}
    last["seconds"] = time.perf_counter() - t0
    return last


def _split_like(model: nn.Module, flat: np.ndarray):
    out, offset = [], 0
    for p in model.parameters():
        n = p.numel()
        out.append(np.asarray(flat[offset:offset + n], dtype=np.float32).reshape(tuple(p.shape)))
        offset += n
    return out


@torch.no_grad()
def evaluate(model: nn.Module, dataset: Dataset, *, batch_size: int, device: torch.device) -> Dict[str, float]:
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses, dices, ious, n = [], [], [], 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(float(segmentation_loss(logits, y)) * len(x))
        d, i = dice_and_iou(logits, y)
        dices.extend(d.tolist())
        ious.extend(i.tolist())
        n += len(x)
    return {"loss": float(sum(losses) / max(n, 1)), "dice": float(np.mean(dices)),
            "iou": float(np.mean(ious)), "n": n}
