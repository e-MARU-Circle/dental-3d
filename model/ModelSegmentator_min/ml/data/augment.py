"""Augmentation utilities for point cloud data."""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, Any


def random_flip_lr(points: np.ndarray, normals: np.ndarray | None = None, prob: float = 0.5) -> tuple[np.ndarray, np.ndarray | None]:
    """Random left-right flip along X axis."""
    if np.random.rand() > prob:
        return points, normals
    P = points.copy()
    P[:, 0] *= -1.0
    N = None if normals is None else normals.copy()
    if N is not None:
        N[:, 0] *= -1.0
    return P, N


def random_rotate_z(points: np.ndarray, normals: np.ndarray | None = None,
                    max_angle: float = 360.0) -> tuple[np.ndarray, np.ndarray | None]:
    """Random rotation around Z axis.

    Args:
        points: (N, 3) point coordinates
        normals: (N, 3) normal vectors (optional)
        max_angle: Maximum rotation angle in degrees (default 360 = full rotation)
    """
    angle_rad = float(np.random.uniform(0.0, max_angle * np.pi / 180.0))
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    P = points @ R.T
    N = None if normals is None else (normals @ R.T)
    return P, N


def random_jitter(points: np.ndarray, sigma: float = 0.005, clip: float = 0.02) -> np.ndarray:
    """Add Gaussian jitter to points."""
    if sigma <= 0:
        return points
    noise = np.random.normal(0.0, sigma, size=points.shape).astype(np.float32)
    if clip > 0:
        noise = np.clip(noise, -clip, clip)
    return points + noise


def random_scale(points: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Random uniform scaling.

    Args:
        points: (N, 3) point coordinates
        scale_range: (min_scale, max_scale) tuple
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return points * scale


def random_anisotropic_scale(points: np.ndarray,
                             scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Random anisotropic scaling (different scale per axis).

    Args:
        points: (N, 3) point coordinates
        scale_range: (min_scale, max_scale) tuple applied per axis
    """
    scales = np.random.uniform(scale_range[0], scale_range[1], size=3).astype(np.float32)
    return points * scales


def random_point_dropout(points: np.ndarray, labels: np.ndarray,
                         feats: Optional[np.ndarray] = None,
                         dropout_ratio: float = 0.05,
                         min_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Randomly drop points from the point cloud.

    Args:
        points: (N, 3) point coordinates
        labels: (N,) labels
        feats: (N, F) features (optional)
        dropout_ratio: Fraction of points to drop
        min_points: Minimum number of points to keep
    """
    N = points.shape[0]
    if dropout_ratio <= 0 or N <= min_points:
        return points, labels, feats

    keep_count = max(min_points, int(N * (1 - dropout_ratio)))
    indices = np.random.choice(N, size=keep_count, replace=False)
    indices = np.sort(indices)

    new_feats = feats[indices] if feats is not None else None
    return points[indices], labels[indices], new_feats


def random_shift(points: np.ndarray, shift_range: float = 0.1) -> np.ndarray:
    """Random translation shift.

    Args:
        points: (N, 3) point coordinates
        shift_range: Maximum shift distance per axis
    """
    shifts = np.random.uniform(-shift_range, shift_range, size=3).astype(np.float32)
    return points + shifts


def random_rotate_perturbation(points: np.ndarray, normals: np.ndarray | None = None,
                                angle_sigma: float = 0.03,
                                angle_clip: float = 0.09) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Small random rotation perturbation for all axes.

    Useful for adding rotational noise while preserving general orientation.
    """
    angles = np.random.normal(0.0, angle_sigma, size=3)
    angles = np.clip(angles, -angle_clip, angle_clip)

    # Rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ], dtype=np.float32)

    Ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ], dtype=np.float32)

    Rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    R = Rz @ Ry @ Rx
    P = points @ R.T
    N = normals @ R.T if normals is not None else None
    return P, N


def elastic_deformation(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    sigma: float = 0.05,
    grid_size: int = 4,
) -> tuple[np.ndarray, np.ndarray | None]:
    """3D elastic deformation using random displacement grid.

    Generates random displacements on a coarse grid and interpolates
    to all points via trilinear interpolation. Naturally warps tooth
    shapes while preserving topology.

    Args:
        points: (N, 3) point coordinates
        normals: (N, 3) normal vectors (optional)
        sigma: displacement magnitude as fraction of bounding box
        grid_size: control grid resolution per axis
    """
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    bbox = mx - mn
    bbox = np.maximum(bbox, 1e-6)

    # Random displacement field on coarse grid
    g = grid_size
    disp = np.random.randn(g, g, g, 3).astype(np.float32) * sigma

    # Normalize point positions to [0, g-1] for interpolation
    norm_pts = (points - mn) / bbox * (g - 1)
    norm_pts = np.clip(norm_pts, 0, g - 1 - 1e-6)

    # Trilinear interpolation
    i0 = norm_pts.astype(np.int32)
    i1 = np.minimum(i0 + 1, g - 1)
    frac = norm_pts - i0  # (N, 3) fractional parts

    fx, fy, fz = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]

    # 8 corner weights
    w000 = (1 - fx) * (1 - fy) * (1 - fz)
    w001 = (1 - fx) * (1 - fy) * fz
    w010 = (1 - fx) * fy * (1 - fz)
    w011 = (1 - fx) * fy * fz
    w100 = fx * (1 - fy) * (1 - fz)
    w101 = fx * (1 - fy) * fz
    w110 = fx * fy * (1 - fz)
    w111 = fx * fy * fz

    ix0, iy0, iz0 = i0[:, 0], i0[:, 1], i0[:, 2]
    ix1, iy1, iz1 = i1[:, 0], i1[:, 1], i1[:, 2]

    d = (w000 * disp[ix0, iy0, iz0] + w001 * disp[ix0, iy0, iz1] +
         w010 * disp[ix0, iy1, iz0] + w011 * disp[ix0, iy1, iz1] +
         w100 * disp[ix1, iy0, iz0] + w101 * disp[ix1, iy0, iz1] +
         w110 * disp[ix1, iy1, iz0] + w111 * disp[ix1, iy1, iz1])

    # Scale displacement by bounding box
    P = points + d * bbox

    if normals is not None:
        return P.astype(np.float32), normals.copy()
    return P.astype(np.float32), None


def random_cutout_sphere(
    points: np.ndarray,
    normals: np.ndarray | None,
    sem2: np.ndarray,
    boundary: np.ndarray,
    boundary_w: np.ndarray,
    n_spheres: int = 1,
    radius_range: Tuple[float, float] = (0.05, 0.15),
    min_keep: int = 1000,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Remove points inside random spheres (spatial CutOut for 3D).

    Args:
        points: (N, 3)
        normals: (N, 3) or None
        sem2, boundary, boundary_w: per-point label arrays
        n_spheres: number of cutout spheres
        radius_range: (min, max) radius as fraction of bbox diagonal
        min_keep: minimum points to retain
    """
    N = points.shape[0]
    if N <= min_keep:
        return points, normals, sem2, boundary, boundary_w

    mn = points.min(axis=0)
    mx = points.max(axis=0)
    diag = np.linalg.norm(mx - mn)
    if diag < 1e-6:
        return points, normals, sem2, boundary, boundary_w

    keep = np.ones(N, dtype=bool)
    for _ in range(n_spheres):
        r = np.random.uniform(radius_range[0], radius_range[1]) * diag
        center_idx = np.random.randint(N)
        center = points[center_idx]
        dists = np.linalg.norm(points - center, axis=1)
        keep &= (dists > r)

    # Ensure minimum points
    if keep.sum() < min_keep:
        return points, normals, sem2, boundary, boundary_w

    P = points[keep]
    Nm = normals[keep] if normals is not None else None
    return P, Nm, sem2[keep], boundary[keep], boundary_w[keep]


def apply_augmentation(points: np.ndarray, labels: np.ndarray,
                        normals: Optional[np.ndarray] = None,
                        feats: Optional[np.ndarray] = None,
                        config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Apply a sequence of augmentations based on config.

    Args:
        points: (N, 3) point coordinates
        labels: (N,) labels
        normals: (N, 3) normal vectors (optional)
        feats: (N, F) additional features (optional)
        config: Augmentation configuration dict

    Returns:
        Tuple of (augmented_points, labels, normals, feats)
    """
    if config is None:
        config = {}

    P = points.copy()
    N = normals.copy() if normals is not None else None
    L = labels.copy()
    F = feats.copy() if feats is not None else None

    # Random flip
    if config.get("flip", False):
        P, N = random_flip_lr(P, N, prob=0.5)

    # Random rotation around Z
    if config.get("rotate", False):
        max_angle = float(config.get("rotate_range", 360))
        P, N = random_rotate_z(P, N, max_angle=max_angle)

    # Random rotation perturbation (all axes)
    if config.get("rotate_perturbation", False):
        sigma = float(config.get("rotate_sigma", 0.03))
        clip = float(config.get("rotate_clip", 0.09))
        P, N = random_rotate_perturbation(P, N, angle_sigma=sigma, angle_clip=clip)

    # Random scale
    if config.get("scale", False):
        scale_range = config.get("scale_range", [0.9, 1.1])
        if isinstance(scale_range, list):
            scale_range = tuple(scale_range)
        P = random_scale(P, scale_range=scale_range)

    # Random jitter
    if config.get("jitter", False):
        sigma = float(config.get("jitter_std", 0.005))
        clip = float(config.get("jitter_clip", 0.02))
        P = random_jitter(P, sigma=sigma, clip=clip)

    # Random shift
    if config.get("shift", False):
        shift_range = float(config.get("shift_range", 0.1))
        P = random_shift(P, shift_range=shift_range)

    # Point dropout (modifies labels and feats too)
    if config.get("dropout_points", False):
        dropout_ratio = float(config.get("dropout_ratio", 0.05))
        min_points = int(config.get("dropout_min_points", 1000))
        # Need to handle features
        combined = np.concatenate([P, N] if N is not None else [P], axis=-1)
        if F is not None:
            combined = np.concatenate([combined, F], axis=-1)

        keep_count = max(min_points, int(len(P) * (1 - dropout_ratio)))
        if keep_count < len(P):
            indices = np.random.choice(len(P), size=keep_count, replace=False)
            indices = np.sort(indices)
            P = P[indices]
            L = L[indices]
            if N is not None:
                N = N[indices]
            if F is not None:
                F = F[indices]

    return P, L, N, F
