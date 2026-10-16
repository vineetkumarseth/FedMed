"""Single entry point for datasets.

Privacy boundary: a hospital node calls :func:`load_hospital_data` with *its own id*
only. The federation server never imports this module (``fedmed audit`` checks that).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from ..config import Config
from . import synthetic as syn


class VolumeDataset(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray):
        self.images = torch.from_numpy(images)
        self.masks = torch.from_numpy(masks)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        return self.images[i], self.masks[i]


def _seed(cfg: Config, hospital_id: int, split: str) -> int:
    offset = {"train": 0, "val": 1000, "test": 2000}[split]
    return cfg.data.seed * 10_000 + hospital_id * 100 + offset


def _synthetic(cfg: Config, hospital_id: int) -> Tuple[VolumeDataset, VolumeDataset]:
    profile = syn.HOSPITAL_PROFILES[((hospital_id - 1) % len(syn.HOSPITAL_PROFILES)) + 1]
    d = cfg.data
    xtr, ytr = syn.make_split(d.train_per_hospital, d.volume_size, d.n_channels, profile,
                              _seed(cfg, hospital_id, "train"))
    xva, yva = syn.make_split(d.val_per_hospital, d.volume_size, d.n_channels, profile,
                              _seed(cfg, hospital_id, "val"))
    return VolumeDataset(xtr, ytr), VolumeDataset(xva, yva)


def _brats(cfg: Config, hospital_id: int) -> Tuple[VolumeDataset, VolumeDataset]:
    from . import brats, partition
    d = cfg.data
    if not d.brats_dir:
        raise ValueError("data.brats_dir must be set when data.source == 'brats'")
    n_h = cfg.federation.num_hospitals
    cases = brats.list_cases(d.brats_dir)[: (d.train_per_hospital + d.val_per_hospital) * n_h]
    mine = partition.iid_partition(len(cases), n_h, d.seed)[hospital_id - 1]
    xs, ys = zip(*(brats.load_case(cases[i], d.volume_size, d.n_channels) for i in mine))
    xs, ys = np.stack(xs), np.stack(ys)
    n_val = min(d.val_per_hospital, max(1, len(xs) // 4))
    return VolumeDataset(xs[:-n_val], ys[:-n_val]), VolumeDataset(xs[-n_val:], ys[-n_val:])


def load_hospital_data(cfg: Config, hospital_id: int) -> Tuple[VolumeDataset, VolumeDataset]:
    """Return (train, val) datasets that live on hospital ``hospital_id`` only."""
    if cfg.data.source == "synthetic":
        return _synthetic(cfg, hospital_id)
    if cfg.data.source == "brats":
        return _brats(cfg, hospital_id)
    raise ValueError(f"Unknown data.source '{cfg.data.source}'")


def load_centralized(cfg: Config) -> Tuple[Dataset, Dataset]:
    """Pool every hospital's data. ONLY for the centralized research baseline."""
    trains, vals = zip(*(load_hospital_data(cfg, h) for h in range(1, cfg.federation.num_hospitals + 1)))
    return ConcatDataset(trains), ConcatDataset(vals)


def load_test_data(cfg: Config) -> VolumeDataset:
    """Held-out cases from a pooled scanner profile (visualisation / final test)."""
    d = cfg.data
    x, y = syn.make_split(d.test_cases, d.volume_size, d.n_channels, syn.POOLED_PROFILE,
                          d.seed * 10_000 + 9_999)
    return VolumeDataset(x, y)
