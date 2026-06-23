"""
Hybrid PointNet2 + Point Transformer backbone for dental segmentation.

Architecture:
  - PN2 SetAbstraction for local geometry (stages 1-2)
  - Point Transformer vector attention at bottleneck (256 points)
  - PN2 SetAbstraction for deepest encoding (stage 3)
  - PN2 FeaturePropagation decoder with skip connections

~800K parameters — between Lite (500K) and Full (3.2M).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_backbone import (
    SetAbstraction,
    FeaturePropagation,
    knn_point,
    index_points,
)


class PointTransformerBlock(nn.Module):
    """Point Transformer block with vector attention and position encoding.

    Implements the vector attention mechanism from Point Transformer (Zhao et al., 2021):
      attention = softmax(MLP(Q - K + pos_enc))  # per-channel weights
      output = sum(attention * (V + pos_enc))

    Args:
        dim: Feature dimension (input and output).
        k: Number of KNN neighbors for local attention.
    """

    def __init__(self, dim: int = 128, k: int = 16):
        super().__init__()
        self.k = k
        self.dim = dim

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Position encoding: maps 3D coordinate differences to dim
        self.pos_enc = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        # Attention MLP: maps (Q-K+pos) to per-channel attention weights
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        # Layer norms and FFN
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) point coordinates
            feat: (B, N, dim) point features

        Returns:
            feat_out: (B, N, dim) updated features
        """
        B, N, _ = feat.shape
        k = min(self.k, N)  # Handle case where N < k

        # 1. KNN graph
        _, knn_idx = knn_point(k, xyz, xyz)  # (B, N, k)

        # 2. Q, K, V projections
        q = self.q_proj(feat)                          # (B, N, dim)
        k_feat = self.k_proj(feat)                     # (B, N, dim)
        v_feat = self.v_proj(feat)                     # (B, N, dim)

        # Gather K, V for neighbors
        k_grouped = index_points(k_feat, knn_idx)      # (B, N, k, dim)
        v_grouped = index_points(v_feat, knn_idx)      # (B, N, k, dim)

        # 3. Position encoding from coordinate differences
        xyz_grouped = index_points(xyz, knn_idx)        # (B, N, k, 3)
        pos_diff = xyz_grouped - xyz.unsqueeze(2)       # (B, N, k, 3)
        pos = self.pos_enc(pos_diff)                    # (B, N, k, dim)

        # 4. Vector attention: per-channel weights
        q_expanded = q.unsqueeze(2).expand_as(k_grouped)  # (B, N, k, dim)
        attn_input = q_expanded - k_grouped + pos          # (B, N, k, dim)
        attn_weights = self.attn_mlp(attn_input)           # (B, N, k, dim)
        attn_weights = F.softmax(attn_weights, dim=2)      # softmax over k neighbors

        # 5. Weighted aggregation
        out = (attn_weights * (v_grouped + pos)).sum(dim=2)  # (B, N, dim)

        # 6. Residual + LayerNorm
        feat = self.norm1(feat + out)

        # 7. FFN + Residual + LayerNorm
        feat = self.norm2(feat + self.ffn(feat))

        return feat


class PointNet2PTHybrid(nn.Module):
    """Hybrid PointNet2 + Point Transformer backbone.

    Architecture:
        SA1: N → 1024 points (local geometry, PN2)
        SA2: 1024 → 256 points (local geometry, PN2)
        PT Blocks: 256 points × dim (global context, Point Transformer)
        SA3: 256 → 64 points (deepest encoding, PN2)
        FP3: 64 → 256 (decoder)
        FP2: 256 → 1024 (decoder)
        FP1: 1024 → N (decoder, output)

    Args:
        in_channels: Input feature dimension (including xyz).
        out_dim: Output per-point feature dimension.
        n_pt_blocks: Number of Point Transformer blocks at bottleneck.
        pt_k: Number of KNN neighbors for Point Transformer attention.
    """

    def __init__(
        self,
        in_channels: int = 13,
        out_dim: int = 128,
        n_pt_blocks: int = 2,
        pt_k: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        extra_ch = in_channels - 3 if in_channels > 3 else 0

        # PN2 Encoder - early stages (local geometry)
        self.sa1 = SetAbstraction(
            npoint=1024, radius=2.0, nsample=16,
            in_channel=extra_ch, mlp=[32, 32, 64],
        )
        self.sa2 = SetAbstraction(
            npoint=256, radius=4.0, nsample=16,
            in_channel=64, mlp=[64, 64, 128],
        )

        # Point Transformer blocks at bottleneck (256 points, dim=128)
        self.pt_blocks = nn.ModuleList([
            PointTransformerBlock(dim=128, k=pt_k)
            for _ in range(n_pt_blocks)
        ])

        # PN2 Encoder - deepest stage
        self.sa3 = SetAbstraction(
            npoint=64, radius=8.0, nsample=16,
            in_channel=128, mlp=[128, 128, 256],
        )

        # PN2 Decoder
        self.fp3 = FeaturePropagation(in_channel=256 + 128, mlp=[128, 128])
        self.fp2 = FeaturePropagation(in_channel=128 + 64, mlp=[128, 64])
        self.fp1 = FeaturePropagation(in_channel=64 + extra_ch, mlp=[64, 64, out_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) input point cloud with features.
                First 3 channels are xyz coordinates.

        Returns:
            features: (B, N, out_dim) per-point features.
        """
        B, N, C = x.shape

        xyz = x[:, :, :3].contiguous()
        points = x[:, :, 3:].contiguous() if C > 3 else None

        # Encoder
        l0_xyz, l0_points = xyz, points
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)   # N → 1024
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)   # 1024 → 256

        # Point Transformer at bottleneck (256 points)
        pt_feat = l2_points
        for pt_block in self.pt_blocks:
            pt_feat = pt_block(l2_xyz, pt_feat)
        l2_points = pt_feat  # Enhanced with global context

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)   # 256 → 64

        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, pt_feat, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        return l0_points
