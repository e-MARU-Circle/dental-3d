from __future__ import annotations

import torch
from torch import nn


class HeadClusterLearned(nn.Module):
    """
    Learned clustering head that predicts centroid offsets and log-variance for each point.

    This is a scaffold to integrate a differentiable clustering approach in Stage2.
    The forward pass returns a dictionary with keys:
      - offset: predicted xyz offset per point
      - log_var: log variance per point to control cluster compactness
    The downstream loss / clustering logic will consume these tensors.
    """

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.offset_head = nn.Linear(hidden, 3)
        self.log_var_head = nn.Linear(hidden, 3)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.mlp(feat)
        offset = self.offset_head(x)
        log_var = self.log_var_head(x)
        return {"offset": offset, "log_var": log_var}

