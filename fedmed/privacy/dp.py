"""Differential privacy for client updates: clipping, Gaussian noise, and an RDP accountant.

Threat model (be honest about it): every round, each hospital sends
``w_global + clip(delta, C) + N(0, (sigma*C)^2 I)``. The guarantee is therefore *per
hospital update* (silo-level DP): it bounds what any observer of that hospital's
transmission - including a curious aggregator that somehow got past the encryption -
can learn about whether the hospital's data influenced the update.

With only 3 silos and ~10^5 parameters, strong guarantees (epsilon <~ 10) require noise
that destroys utility; see ``docs/PRIVACY.md`` and ``scripts/privacy_sweep.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def clip_update(delta: np.ndarray, clip_norm: float) -> Tuple[np.ndarray, float, bool]:
    """Scale ``delta`` so ||delta||_2 <= clip_norm. Returns (clipped, original_norm, was_clipped)."""
    norm = float(np.linalg.norm(delta))
    if norm > clip_norm > 0:
        return delta * (clip_norm / norm), norm, True
    return delta, norm, False


def add_gaussian_noise(delta: np.ndarray, clip_norm: float, noise_multiplier: float,
                       rng: np.random.Generator) -> np.ndarray:
    if noise_multiplier <= 0:
        return delta
    noise = rng.normal(0.0, noise_multiplier * clip_norm, size=delta.shape).astype(np.float32)
    return delta + noise


@dataclass
class RDPAccountant:
    """Renyi-DP accountant for the (non-subsampled) Gaussian mechanism.

    Each round every silo participates, so there is no amplification by sampling; the
    RDP of one Gaussian release with noise multiplier sigma is ``alpha / (2 sigma^2)`` and
    composes additively across rounds.
    """
    noise_multiplier: float
    delta: float = 1e-5
    steps: int = 0

    ORDERS = tuple(np.concatenate([np.linspace(1.01, 10, 60), np.arange(11, 257)]).tolist())

    def step(self, n: int = 1) -> None:
        self.steps += n

    def epsilon(self, steps: Optional[int] = None) -> float:
        t = self.steps if steps is None else steps
        if t == 0:
            return 0.0
        if self.noise_multiplier <= 0:
            return math.inf
        best = math.inf
        for a in self.ORDERS:
            rdp = t * a / (2.0 * self.noise_multiplier ** 2)
            eps = rdp + math.log(1.0 / self.delta) / (a - 1.0)
            best = min(best, eps)
        return best
