from __future__ import annotations

import torch
import torch.nn as nn


class HeadSem2(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.mlp(f)


class HeadContrastive(nn.Module):
    """Contrastive embedding head for boundary learning.

    Produces L2-normalized embeddings for SupCon loss.
    Shares backbone features with HeadSem2.
    """
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 64):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden, out_dim)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """(B, N, in_dim) → (B, N, out_dim) L2-normalized."""
        B, N, C = f.shape
        x = self.linear1(f.reshape(B * N, C))
        x = self.relu(self.bn(x))
        x = self.linear2(x).reshape(B, N, -1)
        return torch.nn.functional.normalize(x, p=2, dim=-1)


class HeadEmb(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        x = self.mlp(f)
        return torch.nn.functional.normalize(x, p=2, dim=-1)


class HeadOffset(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.mlp(f)


class HeadType(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, num_types: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, num_types)
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.mlp(f)


class HeadLmkPres(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, num_marks: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, num_marks)
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.mlp(f)  # logits for multi-label


class HeadLmkHeat(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, num_marks: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, num_marks)
        )

    def forward(self, f: torch.Tensor, emb: torch.Tensor, pres: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([f, emb] + ([pres] if pres is not None else []), dim=-1)
        return self.mlp(x)
