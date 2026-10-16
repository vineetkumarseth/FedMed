"""Procedural 3D "MRI" volumes with tumors, so the whole pipeline runs offline.

Each hospital gets its own *scanner profile* (noise level, tumor contrast, lesion size
distribution). That makes the federation genuinely non-IID, which is the regime where
federated learning is hard and worth studying.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class ScannerProfile:
    name: str
    noise_sigma: float        # additive noise level
    tumor_contrast: float     # how bright the lesion is relative to healthy tissue
    tumor_radius: Tuple[float, float]   # lesion radius range, as a fraction of volume size
    bias_field: float         # strength of the smooth intensity inhomogeneity


# Three institutions with deliberately different acquisition characteristics.
HOSPITAL_PROFILES = {
    1: ScannerProfile("Mercy General (3T, low noise)", 0.04, 1.00, (0.12, 0.22), 0.10),
    2: ScannerProfile("St. Aurelia (1.5T, noisier)", 0.10, 0.80, (0.08, 0.16), 0.25),
    3: ScannerProfile("Nordic Neuro (3T, large lesions)", 0.06, 0.90, (0.16, 0.28), 0.15),
}
# Held-out / global distribution used for the visual test set.
POOLED_PROFILE = ScannerProfile("Pooled test set", 0.07, 0.90, (0.10, 0.24), 0.18)


def _grid(size: int):
    ax = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    return np.meshgrid(ax, ax, ax, indexing="ij")


def _smooth_noise(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    n = gaussian_filter(rng.normal(size=(size,) * 3), sigma=sigma).astype(np.float32)
    return n / (np.abs(n).max() + 1e-6)


def make_case(rng: np.random.Generator, size: int, n_channels: int,
              profile: ScannerProfile) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(image[C,D,H,W], mask[1,D,H,W])`` as float32."""
    z, y, x = _grid(size)

    # Brain: a slightly irregular ellipsoid.
    radii = rng.uniform(0.70, 0.88, size=3)
    wobble = _smooth_noise(rng, size, size / 6) * 0.06
    brain = ((z / radii[0]) ** 2 + (y / radii[1]) ** 2 + (x / radii[2]) ** 2 + wobble) < 1.0

    # Tumor: a lobulated blob inside the brain.
    r_lo, r_hi = profile.tumor_radius
    center = rng.uniform(-0.35, 0.35, size=3)
    radius = rng.uniform(r_lo, r_hi) * 2.0
    shape_noise = _smooth_noise(rng, size, size / 8) * 0.35
    dist = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
    tumor = (dist < radius * (1.0 + shape_noise)) & brain
    if tumor.sum() < 20:  # guarantee a visible lesion
        tumor = (dist < radius) & brain
    core = (dist < radius * 0.55 * (1.0 + shape_noise)) & tumor

    # Healthy tissue texture + smooth bias field.
    img = np.zeros((n_channels, size, size, size), dtype=np.float32)
    bias = 1.0 + profile.bias_field * _smooth_noise(rng, size, size / 3) * 2.0
    for c in range(n_channels):
        tissue = 0.55 + 0.20 * gaussian_filter(rng.normal(size=(size,) * 3), sigma=1.5).astype(np.float32) * 4
        vol = np.where(brain, tissue, 0.02).astype(np.float32)
        # Modality-specific lesion appearance: bright edema on channel 0, enhancing core on others.
        lesion_gain = profile.tumor_contrast * (0.9 if c == 0 else 0.7)
        vol = vol + tumor.astype(np.float32) * lesion_gain * 0.8
        vol = vol + core.astype(np.float32) * lesion_gain * (0.5 if c == 0 else 0.9)
        vol = vol * bias
        vol = vol + rng.normal(0.0, profile.noise_sigma, size=vol.shape).astype(np.float32)
        img[c] = vol

    img = _zscore_within_brain(img, brain)
    return img.astype(np.float32), tumor.astype(np.float32)[None]


def _zscore_within_brain(img: np.ndarray, brain: np.ndarray) -> np.ndarray:
    out = np.zeros_like(img)
    for c in range(img.shape[0]):
        vals = img[c][brain]
        mu, sd = float(vals.mean()), float(vals.std() + 1e-6)
        out[c] = np.where(brain, (img[c] - mu) / sd, 0.0)
    return out


def make_split(n_cases: int, size: int, n_channels: int, profile: ScannerProfile,
               seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n_cases):
        x, y = make_case(rng, size, n_channels, profile)
        xs.append(x)
        ys.append(y)
    return np.stack(xs), np.stack(ys)
