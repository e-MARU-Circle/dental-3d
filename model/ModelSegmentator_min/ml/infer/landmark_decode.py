"""Landmark inference for per-tooth point cloud patches.

Given a tooth point cloud (from instance segmentation), predict 6 landmark
positions using the HeadLandmarkVotingV2 model.

Landmark order: [Mesial, Distal, Cusp, InnerPoint, OuterPoint, FacialPoint]
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ml.models.pointnet2_backbone import PointNet2BackboneLite
from ml.models.head_landmark_reg import HeadLandmarkVotingV2, HeadLandmarkVoting

LANDMARK_NAMES = ["Mesial", "Distal", "Cusp", "InnerPoint", "OuterPoint", "FacialPoint"]


def load_landmark_model(
    ckpt_path: str | Path,
    device: torch.device,
    feat_dim: int = 128,
    num_landmarks: int = 6,
) -> Tuple[PointNet2BackboneLite, torch.nn.Module]:
    """Load landmark model from checkpoint.

    Auto-detects v1 (HeadLandmarkVoting) vs v2 (HeadLandmarkVotingV2)
    based on checkpoint keys.
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    head_state = state.get("head_landmark", {})

    backbone = PointNet2BackboneLite(in_channels=3, out_dim=feat_dim).to(device)
    backbone.load_state_dict(state.get("backbone", {}), strict=False)

    # Detect v2 (has Dropout at mlp.2 → Linear at mlp.3) vs v1 (Linear at mlp.2)
    is_v2 = "mlp.3.weight" in head_state
    if is_v2:
        cfg = state.get("config", {}) or {}
        dropout = float(cfg.get("model", {}).get("head_dropout", 0.3))
        hidden = int(cfg.get("model", {}).get("head_hidden_voting", 256))
        head = HeadLandmarkVotingV2(
            in_dim=feat_dim, hidden=hidden,
            num_landmarks=num_landmarks, dropout=dropout,
        ).to(device)
    else:
        head = HeadLandmarkVoting(
            in_dim=feat_dim, hidden=256, num_landmarks=num_landmarks,
        ).to(device)

    head.load_state_dict(head_state, strict=False)
    backbone.eval()
    head.eval()

    version = "v2" if is_v2 else "v1"
    val_mre = state.get("val_mre", "?")
    print(f"[LANDMARK] Loaded {version} (epoch {state.get('epoch', '?')}, val MRE={val_mre}mm)")
    return backbone, head


def predict_landmarks_single(
    points: np.ndarray,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    device: torch.device,
    max_points: int = 2048,
) -> np.ndarray:
    """Predict 6 landmarks for a single tooth point cloud.

    Args:
        points: (N, 3) tooth surface points in original mm coordinates
        backbone: PointNet2 backbone
        head: HeadLandmarkVoting or V2
        device: torch device
        max_points: subsample to this many points

    Returns:
        landmarks: (6, 3) predicted landmark coordinates in mm
    """
    pts = points.astype(np.float32)
    N = pts.shape[0]

    # Subsample/pad to max_points
    if N >= max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        pts_sub = pts[idx]
    else:
        idx = np.random.choice(N, size=max_points, replace=True)
        pts_sub = pts[idx]

    # Normalize (scalar std, same as training)
    std = float(pts_sub.std() + 1e-6)
    pts_norm = torch.from_numpy(pts_sub / std).unsqueeze(0).to(device)

    with torch.no_grad():
        f = backbone(pts_norm)
        pred_coords, _ = head(f, pts_norm)

    # Convert back to mm
    landmarks_mm = pred_coords.squeeze(0).cpu().numpy() * std
    return landmarks_mm


def predict_landmarks_batch(
    tooth_clouds: Dict[int, np.ndarray],
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    device: torch.device,
    max_points: int = 2048,
) -> Dict[int, np.ndarray]:
    """Predict landmarks for multiple tooth instances.

    Args:
        tooth_clouds: {instance_id: (N_i, 3) points in mm}
        backbone, head, device: model components
        max_points: subsample limit per tooth

    Returns:
        {instance_id: (6, 3) landmark coordinates in mm}
    """
    results: Dict[int, np.ndarray] = {}
    for inst_id, points in tooth_clouds.items():
        if points.shape[0] < 10:
            continue
        landmarks = predict_landmarks_single(
            points, backbone, head, device, max_points=max_points,
        )
        results[inst_id] = landmarks
    return results


def extract_tooth_clouds(
    points: np.ndarray,
    inst_labels: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Extract per-tooth point clouds from instance segmentation result.

    Args:
        points: (N, 3) all points
        inst_labels: (N,) instance labels (-1 = background)

    Returns:
        {instance_id: (N_i, 3) tooth points}
    """
    unique_ids = sorted(int(i) for i in np.unique(inst_labels) if i >= 0)
    clouds: Dict[int, np.ndarray] = {}
    for inst_id in unique_ids:
        mask = inst_labels == inst_id
        clouds[inst_id] = points[mask]
    return clouds
