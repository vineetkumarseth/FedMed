"""Loader for the public BraTS dataset (NIfTI files).

Expected layout (BraTS 2021 style)::

    <brats_dir>/BraTS2021_00000/BraTS2021_00000_flair.nii.gz
                               /BraTS2021_00000_t1ce.nii.gz
                               /BraTS2021_00000_seg.nii.gz
                ...

Volumes are z-scored inside the brain, cropped to the brain bounding box and resized
to ``size**3``. Labels are collapsed to a binary "whole tumor" mask (``seg > 0``).

NOTE: this loader has not been run against the real BraTS download in the project's
CI (no data access there). It follows the documented file layout; expect to adjust
file-name patterns if your copy differs.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

MODALITY_SUFFIX = ("flair", "t1ce", "t1", "t2")


def list_cases(root: str | Path) -> List[Path]:
    root = Path(root)
    cases = sorted(p for p in root.iterdir() if p.is_dir())
    if not cases:
        raise FileNotFoundError(f"No case folders found in {root}")
    return cases


def _load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("BraTS loading needs nibabel: pip install 'fedmed[brats]'") from exc
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def _find(case: Path, suffix: str) -> Path:
    matches = sorted(case.glob(f"*_{suffix}.nii*"))
    if not matches:
        raise FileNotFoundError(f"{case.name}: missing '{suffix}' volume")
    return matches[0]


def _resize(vol: np.ndarray, size: int, order: int) -> np.ndarray:
    from scipy.ndimage import zoom
    return zoom(vol, [size / s for s in vol.shape], order=order)


def load_case(case: Path, size: int, n_channels: int) -> Tuple[np.ndarray, np.ndarray]:
    mods: Sequence[str] = MODALITY_SUFFIX[:n_channels]
    chans = [_load_nifti(_find(case, m)) for m in mods]
    seg = _load_nifti(_find(case, "seg")) > 0

    brain = chans[0] > 0
    idx = np.argwhere(brain)
    lo, hi = idx.min(0), idx.max(0) + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))

    out = []
    for vol in chans:
        v, m = vol[sl], brain[sl]
        mu, sd = v[m].mean(), v[m].std() + 1e-6
        out.append(_resize(np.where(m, (v - mu) / sd, 0.0), size, order=1))
    mask = _resize(seg[sl].astype(np.float32), size, order=0)
    return np.stack(out).astype(np.float32), (mask[None] > 0.5).astype(np.float32)
