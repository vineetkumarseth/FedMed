from __future__ import annotations

import torch.nn as nn

from ..config import Config
from .unet3d import UNet3D


def build_model(cfg: Config) -> nn.Module:
    name = cfg.model.name
    if name == "unet3d":
        return UNet3D(cfg.data.n_channels, 1, cfg.model.base_filters, cfg.model.depth)
    if name == "monai_unet":  # optional: MONAI's residual 3D U-Net
        try:
            from monai.networks.nets import UNet
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("model.name='monai_unet' needs MONAI: pip install 'fedmed[monai]'") from exc
        f = cfg.model.base_filters
        return UNet(
            spatial_dims=3, in_channels=cfg.data.n_channels, out_channels=1,
            channels=tuple(f * 2 ** i for i in range(cfg.model.depth + 1)),
            strides=(2,) * cfg.model.depth, num_res_units=1, norm="instance",
        )
    raise ValueError(f"Unknown model.name '{name}'")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
