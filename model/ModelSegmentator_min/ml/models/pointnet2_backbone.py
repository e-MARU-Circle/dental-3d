"""
PointNet++ Backbone for dental segmentation.
Implements hierarchical point cloud processing with Set Abstraction and Feature Propagation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate squared Euclidean distance between each pair of points.

    Args:
        src: (B, N, C) source points
        dst: (B, M, C) destination points

    Returns:
        dist: (B, N, M) squared distance matrix
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest point sampling.

    Args:
        xyz: (B, N, 3) point coordinates
        npoint: number of points to sample

    Returns:
        centroids: (B, npoint) indices of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10

    # Random starting point
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Index points by indices.

    Args:
        points: (B, N, C) input points
        idx: (B, S) or (B, S, K) indices

    Returns:
        indexed_points: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball query - find all points within radius of each centroid.

    Args:
        radius: ball radius
        nsample: max number of samples per ball
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points (centroids)

    Returns:
        group_idx: (B, S, nsample) indices of grouped points
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # If not enough points in ball, replicate first point
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    KNN query.

    Args:
        k: number of neighbors
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points

    Returns:
        dist: (B, S, k) squared distances
        idx: (B, S, k) indices
    """
    sqrdists = square_distance(new_xyz, xyz)
    dist, idx = torch.topk(sqrdists, k, dim=-1, largest=False)
    return dist, idx


class SharedMLP(nn.Module):
    """Shared MLP applied pointwise."""

    def __init__(self, channels: List[int], bn: bool = True, activation: str = 'relu'):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i+1], 1))
            if bn:
                layers.append(nn.BatchNorm1d(channels[i+1]))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, N) input features
        Returns:
            (B, C', N) output features
        """
        return self.mlp(x)


class SetAbstraction(nn.Module):
    """
    Set Abstraction layer for hierarchical point processing.
    Samples points, groups neighbors, and extracts features.
    """

    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: List[int],
        group_all: bool = False,
        use_xyz: bool = True
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.use_xyz = use_xyz

        # MLP for processing grouped features
        in_ch = in_channel + 3 if use_xyz else in_channel
        self.mlp = SharedMLP([in_ch] + mlp)

    def forward(
        self,
        xyz: torch.Tensor,
        points: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3) coordinates
            points: (B, N, C) features (optional)

        Returns:
            new_xyz: (B, npoint, 3) sampled coordinates
            new_points: (B, npoint, D) features
        """
        B, N, _ = xyz.shape

        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz.view(B, 1, N, 3)
            if points is not None:
                grouped_points = points.view(B, 1, N, -1)
        else:
            # Farthest point sampling
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)

            # Ball query
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
            grouped_xyz -= new_xyz.view(B, self.npoint, 1, 3)  # Normalize to local frame

            if points is not None:
                grouped_points = index_points(points, idx)  # (B, npoint, nsample, C)

        # Concatenate xyz coordinates
        if points is not None:
            if self.use_xyz:
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_points
        else:
            new_points = grouped_xyz

        # (B, npoint, nsample, C) -> (B, C, nsample, npoint) -> apply MLP -> (B, D, npoint)
        new_points = new_points.permute(0, 3, 2, 1)  # (B, C, nsample, npoint)
        B, C, K, S = new_points.shape
        new_points = new_points.reshape(B, C, K * S)
        new_points = self.mlp(new_points)
        new_points = new_points.reshape(B, -1, K, S)
        new_points = torch.max(new_points, dim=2)[0]  # Max pooling over neighbors
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, D)

        return new_xyz, new_points


class SetAbstractionMSG(nn.Module):
    """
    Multi-Scale Grouping Set Abstraction.
    Uses multiple radii for multi-scale feature extraction.
    """

    def __init__(
        self,
        npoint: int,
        radius_list: List[float],
        nsample_list: List[int],
        in_channel: int,
        mlp_list: List[List[int]]
    ):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        self.mlps = nn.ModuleList()
        for i in range(len(radius_list)):
            mlp = SharedMLP([in_channel + 3] + mlp_list[i])
            self.mlps.append(mlp)

    def forward(
        self,
        xyz: torch.Tensor,
        points: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3) coordinates
            points: (B, N, C) features

        Returns:
            new_xyz: (B, npoint, 3)
            new_points: (B, npoint, sum(mlp_out_dims))
        """
        B, N, _ = xyz.shape

        # Farthest point sampling
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)

        new_points_list = []
        for i, (radius, nsample, mlp) in enumerate(zip(self.radius_list, self.nsample_list, self.mlps)):
            # Ball query
            idx = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz -= new_xyz.view(B, self.npoint, 1, 3)

            if points is not None:
                grouped_points = index_points(points, idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz

            # Apply MLP
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # (B, C, nsample, npoint)
            B_cur, C_cur, K, S = grouped_points.shape
            grouped_points = grouped_points.reshape(B_cur, C_cur, K * S)
            grouped_points = mlp(grouped_points)
            grouped_points = grouped_points.reshape(B_cur, -1, K, S)
            grouped_points = torch.max(grouped_points, dim=2)[0]  # (B, D, npoint)
            grouped_points = grouped_points.permute(0, 2, 1)  # (B, npoint, D)

            new_points_list.append(grouped_points)

        new_points = torch.cat(new_points_list, dim=-1)
        return new_xyz, new_points


class FeaturePropagation(nn.Module):
    """
    Feature Propagation layer for upsampling.
    Interpolates features from coarse level to fine level.
    """

    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        self.mlp = SharedMLP([in_channel] + mlp)

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: Optional[torch.Tensor],
        points2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            xyz1: (B, N, 3) fine-level coordinates
            xyz2: (B, S, 3) coarse-level coordinates
            points1: (B, N, D1) fine-level features (skip connection)
            points2: (B, S, D2) coarse-level features to interpolate

        Returns:
            new_points: (B, N, D) interpolated features
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # Global feature, broadcast to all points
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # Distance-weighted interpolation (inverse distance weighting)
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # Use 3 nearest neighbors

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1),
                dim=2
            )

        # Concatenate with skip connection
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        # Apply MLP
        new_points = new_points.permute(0, 2, 1)  # (B, C, N)
        new_points = self.mlp(new_points)
        new_points = new_points.permute(0, 2, 1)  # (B, N, D)

        return new_points


class PointNet2Backbone(nn.Module):
    """
    PointNet++ backbone for semantic segmentation.

    Architecture:
    - 4 Set Abstraction layers for hierarchical encoding
    - 4 Feature Propagation layers for decoding
    - Skip connections between encoder and decoder
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_dim: int = 128,
        use_normals: bool = True,
        use_msg: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.use_normals = use_normals

        # Additional input features (beyond xyz)
        extra_ch = in_channels - 3 if in_channels > 3 else 0

        if use_msg:
            # Multi-Scale Grouping
            self.sa1 = SetAbstractionMSG(
                npoint=4096,
                radius_list=[0.5, 1.0, 2.0],
                nsample_list=[16, 32, 64],
                in_channel=extra_ch,
                mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
            )
            self.sa2 = SetAbstractionMSG(
                npoint=1024,
                radius_list=[1.0, 2.0, 4.0],
                nsample_list=[16, 32, 64],
                in_channel=64+128+128,
                mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]]
            )
            sa2_out = 128 + 256 + 256
        else:
            # Single Scale Grouping
            self.sa1 = SetAbstraction(
                npoint=4096, radius=1.0, nsample=32,
                in_channel=extra_ch, mlp=[64, 64, 128]
            )
            self.sa2 = SetAbstraction(
                npoint=1024, radius=2.0, nsample=32,
                in_channel=128, mlp=[128, 128, 256]
            )
            sa2_out = 256

        self.sa3 = SetAbstraction(
            npoint=256, radius=4.0, nsample=32,
            in_channel=sa2_out, mlp=[256, 256, 512]
        )
        self.sa4 = SetAbstraction(
            npoint=64, radius=8.0, nsample=32,
            in_channel=512, mlp=[512, 512, 1024]
        )

        # Feature Propagation (decoder)
        self.fp4 = FeaturePropagation(in_channel=1024 + 512, mlp=[512, 512])
        self.fp3 = FeaturePropagation(in_channel=512 + sa2_out, mlp=[512, 256])
        self.fp2 = FeaturePropagation(in_channel=256 + 128 if not use_msg else 256 + 320, mlp=[256, 128])
        self.fp1 = FeaturePropagation(in_channel=128 + extra_ch, mlp=[128, 128, out_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) input point cloud with features
                First 3 channels are xyz coordinates

        Returns:
            features: (B, N, out_dim) per-point features
        """
        B, N, C = x.shape

        # Split xyz and features
        xyz = x[:, :, :3].contiguous()
        if C > 3:
            points = x[:, :, 3:].contiguous()
        else:
            points = None

        # Encoder (downsampling)
        l0_xyz = xyz
        l0_points = points

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Decoder (upsampling with skip connections)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        return l0_points


class PointNet2BackboneLite(nn.Module):
    """
    Lightweight PointNet++ backbone.
    Fewer layers and smaller dimensions for faster inference.
    Memory efficient version for limited GPU/MPS.
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_dim: int = 128
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        extra_ch = in_channels - 3 if in_channels > 3 else 0

        # Encoder (3 levels) - reduced npoint and nsample for memory efficiency
        self.sa1 = SetAbstraction(
            npoint=1024, radius=2.0, nsample=16,
            in_channel=extra_ch, mlp=[32, 32, 64]
        )
        self.sa2 = SetAbstraction(
            npoint=256, radius=4.0, nsample=16,
            in_channel=64, mlp=[64, 64, 128]
        )
        self.sa3 = SetAbstraction(
            npoint=64, radius=8.0, nsample=16,
            in_channel=128, mlp=[128, 128, 256]
        )

        # Decoder
        self.fp3 = FeaturePropagation(in_channel=256 + 128, mlp=[128, 128])
        self.fp2 = FeaturePropagation(in_channel=128 + 64, mlp=[128, 64])
        self.fp1 = FeaturePropagation(in_channel=64 + extra_ch, mlp=[64, 64, out_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) input features
        Returns:
            (B, N, out_dim) per-point features
        """
        B, N, C = x.shape

        xyz = x[:, :, :3].contiguous()
        points = x[:, :, 3:].contiguous() if C > 3 else None

        l0_xyz, l0_points = xyz, points
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        return l0_points


# For backward compatibility
def get_backbone(name: str = 'pointnet2', **kwargs) -> nn.Module:
    """Factory function to get backbone by name."""
    if name == 'pointnet2':
        return PointNet2Backbone(**kwargs)
    elif name == 'pointnet2_lite':
        return PointNet2BackboneLite(**kwargs)
    elif name == 'dummy':
        from .backbone_pointnet2 import DummyBackbone
        return DummyBackbone(**kwargs)
    else:
        raise ValueError(f"Unknown backbone: {name}")
