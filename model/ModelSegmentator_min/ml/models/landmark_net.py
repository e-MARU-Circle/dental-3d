"""
Improved Landmark Detection Network.
Uses PointNet++ backbone with attention-based landmark regression.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class PointNetEncoderImproved(nn.Module):
    """Improved PointNet encoder with more capacity and skip connections."""

    def __init__(self, input_dim: int = 3, feature_dim: int = 512):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.mlp4 = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, C) input points

        Returns:
            global_features: (B, feature_dim) global features
            point_features: (B, N, feature_dim) per-point features
        """
        B, N, C = x.shape

        # Reshape for BatchNorm1d
        x = x.view(B * N, C)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = x.view(B, N, -1)

        # Global max pooling
        global_feat = torch.max(x, dim=1).values  # (B, feature_dim)

        return global_feat, x


class SpatialAttention(nn.Module):
    """Spatial attention for landmark localization."""

    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert self.head_dim * num_heads == feature_dim

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.out = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) point features
            coords: (B, N, 3) point coordinates

        Returns:
            attended: (B, N, D) attended features
        """
        B, N, D = x.shape

        # Add positional encoding from coordinates
        pos_enc = torch.sin(coords * 10.0)  # Simple positional encoding
        x = x + pos_enc.repeat(1, 1, D // 3 + 1)[:, :, :D]

        q = self.query(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        return self.out(out)


class LandmarkQueryDecoder(nn.Module):
    """
    Query-based landmark decoder.
    Uses learnable queries to extract landmark-specific features.
    """

    def __init__(self, num_landmarks: int, feature_dim: int = 512, num_layers: int = 3):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.feature_dim = feature_dim

        # Learnable landmark queries
        self.landmark_queries = nn.Parameter(torch.randn(num_landmarks, feature_dim))

        # Cross-attention layers
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
            for _ in range(num_layers)
        ])
        self.self_attention = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim * 2, feature_dim)
            )
            for _ in range(num_layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])

        # Output heads
        self.coord_head = nn.Linear(feature_dim, 3)
        self.confidence_head = nn.Linear(feature_dim, 1)

    def forward(
        self,
        point_features: torch.Tensor,
        global_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            point_features: (B, N, D) per-point features
            global_features: (B, D) global features

        Returns:
            coords: (B, num_landmarks, 3) predicted landmark coordinates
            confidence: (B, num_landmarks) confidence scores
        """
        B = point_features.shape[0]

        # Initialize queries with learnable embeddings + global context
        queries = self.landmark_queries.unsqueeze(0).expand(B, -1, -1)  # (B, L, D)
        queries = queries + global_features.unsqueeze(1)  # Add global context

        for i in range(len(self.cross_attention)):
            # Self-attention among queries
            q_attn, _ = self.self_attention[i](queries, queries, queries)
            queries = self.norm1[i](queries + q_attn)

            # Cross-attention to point features
            cross_attn, _ = self.cross_attention[i](queries, point_features, point_features)
            queries = self.norm2[i](queries + cross_attn)

            # FFN
            queries = self.norm3[i](queries + self.ffn[i](queries))

        # Output
        coords = self.coord_head(queries)  # (B, L, 3)
        confidence = torch.sigmoid(self.confidence_head(queries).squeeze(-1))  # (B, L)

        return coords, confidence


class HeatmapDecoder(nn.Module):
    """
    Heatmap-based landmark decoder.
    Predicts per-point heatmap for each landmark, then extracts coordinates.
    """

    def __init__(self, num_landmarks: int, feature_dim: int = 512):
        super().__init__()
        self.num_landmarks = num_landmarks

        self.heatmap_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_landmarks)
        )

    def forward(
        self,
        point_features: torch.Tensor,
        coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            point_features: (B, N, D) per-point features
            coords: (B, N, 3) point coordinates

        Returns:
            predicted_coords: (B, num_landmarks, 3) landmark coordinates
            heatmaps: (B, N, num_landmarks) per-point heatmaps
        """
        B, N, D = point_features.shape

        # Predict heatmaps
        heatmaps = self.heatmap_head(point_features)  # (B, N, L)
        heatmaps = F.softmax(heatmaps, dim=1)  # Softmax over points

        # Soft-argmax to get coordinates
        # weighted sum of coordinates
        predicted_coords = torch.einsum('bnl,bnc->blc', heatmaps, coords)  # (B, L, 3)

        return predicted_coords, heatmaps


class ImprovedLandmarkNet(nn.Module):
    """
    Improved Landmark Detection Network.

    Features:
    - Better encoder with skip connections
    - Query-based or heatmap-based decoding
    - Confidence prediction
    - Support for variable number of landmarks
    """

    def __init__(
        self,
        num_landmarks: int,
        input_dim: int = 3,
        feature_dim: int = 512,
        decoder_type: str = 'query',  # 'query' or 'heatmap'
        use_attention: bool = True
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.decoder_type = decoder_type

        # Encoder
        self.encoder = PointNetEncoderImproved(input_dim=input_dim, feature_dim=feature_dim)

        # Optional attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpatialAttention(feature_dim, num_heads=4)

        # Decoder
        if decoder_type == 'query':
            self.decoder = LandmarkQueryDecoder(num_landmarks, feature_dim)
        else:
            self.decoder = HeatmapDecoder(num_landmarks, feature_dim)

    def forward(
        self,
        points: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            points: (B, N, 3) input point cloud

        Returns:
            Dictionary containing:
            - coords: (B, num_landmarks, 3) predicted coordinates
            - confidence: (B, num_landmarks) confidence scores (query decoder only)
            - heatmaps: (B, N, num_landmarks) heatmaps (heatmap decoder only)
        """
        B, N, C = points.shape

        # Encode
        global_feat, point_feat = self.encoder(points)

        # Attention
        if self.use_attention:
            point_feat = point_feat + self.attention(point_feat, points)

        # Decode
        if self.decoder_type == 'query':
            coords, confidence = self.decoder(point_feat, global_feat)
            return {
                'coords': coords,
                'confidence': confidence
            }
        else:
            coords, heatmaps = self.decoder(point_feat, points)
            return {
                'coords': coords,
                'heatmaps': heatmaps
            }


class WingLoss(nn.Module):
    """
    Wing Loss for landmark detection.
    Better than MSE for small errors, more robust for large errors.
    """

    def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = omega - omega * torch.log(torch.tensor(1.0 + omega / epsilon))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1.0 + diff / self.epsilon),
            diff - self.C.to(diff.device)
        )

        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            return loss.sum() / (mask.sum() * 3 + 1e-6)
        return loss.mean()


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss - improves on Wing Loss for face/dental landmarks.
    """

    def __init__(self, omega: float = 14.0, theta: float = 0.5, epsilon: float = 1.0, alpha: float = 2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = torch.abs(pred - target)

        A = self.omega * (1 / (1 + (self.theta / self.epsilon) ** (self.alpha - target.new_tensor(1.0)))) * \
            (self.alpha - 1) * ((self.theta / self.epsilon) ** (self.alpha - 2)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + (self.theta / self.epsilon) ** (self.alpha - 1))

        loss = torch.where(
            diff < self.theta,
            self.omega * torch.log(1 + (diff / self.epsilon) ** (self.alpha - 1)),
            A * diff - C
        )

        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            return loss.sum() / (mask.sum() * 3 + 1e-6)
        return loss.mean()


def create_landmark_model(
    num_landmarks: int,
    model_type: str = 'improved',
    **kwargs
) -> nn.Module:
    """Factory function to create landmark detection models."""

    if model_type == 'improved':
        return ImprovedLandmarkNet(num_landmarks=num_landmarks, **kwargs)
    elif model_type == 'simple':
        # Simple regression model for backward compatibility
        from ml.train_landmarks import LandmarkRegressor
        return LandmarkRegressor(num_landmarks=num_landmarks)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
