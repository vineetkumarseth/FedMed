from __future__ import annotations

import torch


@torch.no_grad()
def dice_and_iou(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """Per-case Dice and IoU (tensors of shape [B])."""
    pred = (torch.sigmoid(logits) > 0.5).float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dims)
    p, t = pred.sum(dims), target.sum(dims)
    dice = (2 * inter + eps) / (p + t + eps)
    iou = (inter + eps) / (p + t - inter + eps)
    return dice, iou
