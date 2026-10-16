"""Ways to split a pool of cases across hospitals (used for real datasets like BraTS)."""
from __future__ import annotations

from typing import List

import numpy as np


def iid_partition(n_items: int, n_parts: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.sort(p) for p in np.array_split(rng.permutation(n_items), n_parts)]


def lesion_size_partition(tumor_volumes: np.ndarray, n_parts: int, alpha: float,
                          seed: int) -> List[np.ndarray]:
    """Non-IID split: Dirichlet allocation over lesion-size quantile bins.

    Small ``alpha`` => each hospital specialises in a different lesion-size regime.
    """
    rng = np.random.default_rng(seed)
    bins = np.digitize(tumor_volumes, np.quantile(tumor_volumes, [1 / 3, 2 / 3]))
    parts: List[List[int]] = [[] for _ in range(n_parts)]
    for b in np.unique(bins):
        members = rng.permutation(np.where(bins == b)[0])
        shares = rng.dirichlet(alpha * np.ones(n_parts))
        cuts = (np.cumsum(shares) * len(members)).astype(int)[:-1]
        for part, chunk in zip(parts, np.split(members, cuts)):
            part.extend(chunk.tolist())
    assert sum(len(p) for p in parts) == len(tumor_volumes)
    return [np.sort(np.array(p, dtype=int)) for p in parts]
