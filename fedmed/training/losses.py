from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice + BCE: Dice handles the extreme foreground/background imbalance, BCE keeps gradients stable."""
    return soft_dice_loss(logits, target) + F.binary_cross_entropy_with_logits(logits, target)
