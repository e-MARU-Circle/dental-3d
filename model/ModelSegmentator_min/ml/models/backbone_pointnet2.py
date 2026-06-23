"""Backbone placeholder (PointNet2-Lite style)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DummyBackbone(nn.Module):
    def __init__(self, in_channels: int = 12, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,F)
        return self.net(x)

