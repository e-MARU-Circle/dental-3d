"""Landmark regression heads: regression, heatmap, and offset voting."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadLandmarkReg(nn.Module):
    """Predict landmark coordinates via global pooling + MLP.

    Input:  per-point features (B, N, D) from backbone
    Output: landmark coords   (B, num_landmarks, 3)
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden: int = 256,
        num_landmarks: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_landmarks * 3),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: (B, N, D) per-point features
        Returns:
            (B, num_landmarks, 3) predicted coordinates
        """
        global_feat = torch.max(f, dim=1).values  # (B, D)
        out = self.mlp(global_feat)  # (B, num_landmarks * 3)
        return out.view(-1, self.num_landmarks, 3)


class HeadLandmarkHeatmap(nn.Module):
    """Heatmap-based landmark regression: per-point softmax + soft-argmax.

    Input:  per-point features (B, N, D) from backbone
            point coordinates  (B, N, 3)
    Output: landmark coords    (B, num_landmarks, 3)
            heatmaps           (B, N, num_landmarks)
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden: int = 128,
        num_landmarks: int = 6,
        tau: float = 1.0,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.tau = tau  # mutable: training loop anneals this
        self.heatmap_head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_landmarks),
        )

    def forward(
        self, f: torch.Tensor, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f:      (B, N, D) per-point features from backbone
            coords: (B, N, 3) point xyz coordinates
        Returns:
            pred_coords: (B, num_landmarks, 3)
            heatmaps:    (B, N, num_landmarks) softmax-normalized
        """
        logits = self.heatmap_head(f)  # (B, N, L)
        heatmaps = F.softmax(logits / self.tau, dim=1)  # softmax over N points
        pred_coords = torch.einsum('bnl,bnc->blc', heatmaps, coords)  # (B, L, 3)
        return pred_coords, heatmaps


class HeadLandmarkVoting(nn.Module):
    """Offset voting landmark regression: each point votes for landmark positions.

    Each point predicts an offset to each landmark. The final landmark position
    is the inverse-distance-weighted mean of all votes.

    Input:  per-point features (B, N, D) from backbone
            point coordinates  (B, N, 3)
    Output: landmark coords    (B, num_landmarks, 3)
            offsets            (B, N, num_landmarks, 3)
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden: int = 256,
        num_landmarks: int = 6,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_landmarks * 3),
        )

    def forward(
        self, f: torch.Tensor, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = f.shape
        L = self.num_landmarks
        offsets = self.mlp(f).view(B, N, L, 3)
        votes = coords.unsqueeze(2) + offsets
        weights = 1.0 / (offsets.norm(dim=-1) + 0.1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        pred_coords = (votes * weights.unsqueeze(-1)).sum(dim=1)
        return pred_coords, offsets


class HeadLandmarkVotingV2(nn.Module):
    """v2: HeadLandmarkVoting + Dropout for regularization.

    MRE: 4.23mm (v1) → 2.96mm (v2, -30%).
    Checkpoint: landmark_vote_v2_best.pth
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden: int = 256,
        num_landmarks: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_landmarks * 3),
        )

    def forward(
        self, f: torch.Tensor, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f:      (B, N, D) per-point features from backbone
            coords: (B, N, 3) point xyz coordinates
        Returns:
            pred_coords: (B, num_landmarks, 3) weighted-mean landmark positions
            offsets:     (B, N, num_landmarks, 3) per-point offsets
        """
        B, N, _ = f.shape
        L = self.num_landmarks
        offsets = self.mlp(f).view(B, N, L, 3)  # (B, N, L, 3)

        # votes: each point's prediction for each landmark
        votes = coords.unsqueeze(2) + offsets  # (B, N, L, 3)

        # inverse-distance weighting: short offsets = high confidence
        weights = 1.0 / (offsets.norm(dim=-1) + 0.1)  # (B, N, L)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # normalize

        # weighted mean aggregation
        pred_coords = (votes * weights.unsqueeze(-1)).sum(dim=1)  # (B, L, 3)

        return pred_coords, offsets
