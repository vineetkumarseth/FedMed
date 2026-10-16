"""Export segmentation overlays for the dashboard.

Loads a hospital's decrypted copy of the final global model, predicts on held-out cases
and writes ``segmentation.json`` (volumes packed as base64 uint8) for the viewer.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import Config
from .data import load_test_data
from .models import build_model
from .training.local import resolve_device


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode("ascii")


def export_segmentations(cfg: Config, run_dir: Path, model_path: Optional[Path] = None, n_cases: int = 4) -> Path:
    run_dir = Path(run_dir)
    model_path = Path(model_path) if model_path else run_dir / "models" / "hospital_1.pt"
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    device = resolve_device(cfg.train.device)
    model.to(device).eval()

    ds = load_test_data(cfg)
    cases = []
    with torch.no_grad():
        for i in range(min(n_cases, len(ds))):
            x, y = ds[i]
            pred = (torch.sigmoid(model(x[None].to(device)))[0, 0] > 0.5).cpu().numpy()
            gt = y[0].numpy() > 0.5
            inter = float((pred & gt).sum())
            dice = (2 * inter + 1e-6) / (float(pred.sum() + gt.sum()) + 1e-6)
            img = x[0].numpy()
            lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
            img8 = (np.clip((img - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
            zs = np.where(gt.any(axis=(1, 2)))[0]
            cases.append({"id": f"case-{i + 1:02d}", "dice": round(dice, 4), "tumor_voxels": int(gt.sum()),
                          "focus_slice": int(zs[len(zs) // 2]) if len(zs) else int(img.shape[0] // 2),
                          "image": _b64(img8), "gt": _b64(gt), "pred": _b64(pred)})
    out = {"size": int(cfg.data.volume_size), "round": int(ckpt.get("round", 0)),
           "model": model_path.name, "cases": cases}
    path = run_dir / "segmentation.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    return path
