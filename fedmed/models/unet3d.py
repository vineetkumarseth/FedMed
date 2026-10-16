"""A compact 3D U-Net built for federated learning.

Design choices that matter in a federation:

* **InstanceNorm instead of BatchNorm.** BatchNorm keeps running statistics that are
  not parameters, differ per hospital, and average badly. InstanceNorm has no running
  state, so the model is *exactly* its parameter list and aggregation is well defined.
* No dropout / stochastic buffers, so a client's local update is reproducible from its seed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c_out, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c_out, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1,
                 base_filters: int = 8, depth: int = 3):
        super().__init__()
        chans = [base_filters * 2 ** i for i in range(depth + 1)]
        self.encoders = nn.ModuleList()
        c_prev = in_channels
        for c in chans[:-1]:
            self.encoders.append(ConvBlock(c_prev, c))
            c_prev = c
        self.bottleneck = ConvBlock(chans[-2], chans[-1])
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for c_hi, c_lo in zip(reversed(chans[1:]), reversed(chans[:-1])):
            self.upconvs.append(nn.ConvTranspose3d(c_hi, c_lo, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(c_lo * 2, c_lo))
        self.head = nn.Conv3d(chans[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = F.max_pool3d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)  # logits
