"""Feature preprocessing（Stage-1向けの幾何特徴を追加）。

出力する特徴（N,F）:
- 座標 (x,y,z)
- 法線 (nx,ny,nz)（無い場合は0）
- 近傍幾何特徴（kNNで推定）
  - normal_var: 近傍法線の分散指標
  - density: 近傍までの平均距離（小さいほど高密度）
  - PCA固有値からの派生量（線形性/平面性/球面性/異方性/オムニバリアンス）

注: SciPyがあれば cKDTree を利用、無ければ簡易フォールバック（計算コストを抑えるため最小限）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class FeatureConfig:
    knn: int = 32
    radius_mm: float = 1.0
    add_curvature: bool = True
    add_normal_var: bool = True
    add_density: bool = True


def _knn_indices(points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dists, idx) of shape (N,k). Uses SciPy KDTree if available, else brute force in chunks."""
    N = points.shape[0]
    try:
        from scipy.spatial import cKDTree as KDTree  # type: ignore
        tree = KDTree(points)
        d, idx = tree.query(points, k=min(k, N))
        if np.ndim(idx) == 1:
            idx = idx[:, None]
            d = d[:, None]
        return d.astype(np.float32), idx.astype(np.int32)
    except Exception:
        # Brute-force（遅いのでチャンク）。低メモリ向けに小さめのチャンクを使用。
        bs = 1024  # reduce memory footprint (was 4096)
        kk = min(k, N)
        d_all = np.zeros((N, kk), dtype=np.float32)
        i_all = np.zeros((N, kk), dtype=np.int32)
        for i in range(0, N, bs):
            Q = points[i:i+bs]
            dist2 = ((Q[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
            idx = np.argpartition(dist2, kth=kk-1, axis=1)[:, :kk]
            d = np.take_along_axis(dist2, idx, axis=1)
            order = np.argsort(d, axis=1)
            d_sorted = np.take_along_axis(d, order, axis=1)
            idx_sorted = np.take_along_axis(idx, order, axis=1)
            d_all[i:i+bs] = np.sqrt(d_sorted).astype(np.float32)
            i_all[i:i+bs] = idx_sorted.astype(np.int32)
        return d_all, i_all


def _pca_eigs(X: np.ndarray) -> Tuple[float, float, float]:
    """Return eigenvalues (l1>=l2>=l3) of 3x3 covariance matrix of X (M,3)."""
    if X.shape[0] < 3:
        return 0.0, 0.0, 0.0
    Y = X - X.mean(axis=0, keepdims=True)
    C = (Y.T @ Y) / max(1, Y.shape[0] - 1)
    w = np.linalg.eigvalsh(C)
    w = np.sort(w)[::-1]
    return float(w[0]), float(w[1]), float(w[2])


def _robust_standardize(arr_list: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for a in arr_list:
        mu = np.median(a)
        mad = np.median(np.abs(a - mu)) + 1e-6
        out.append(((a - mu) / (1.4826 * mad)).astype(np.float32))
    return out


def compute_features(points: np.ndarray, normals: np.ndarray | None = None, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Compute per-point features with neighborhood statistics.

    Returns matrix (N, F): [xyz, nxyz, normal_var, density, linearity, planarity, sphericity, anisotropy, omnivariance]
    """
    pts = points.astype(np.float32)
    N = pts.shape[0]
    if normals is None or normals.shape != pts.shape:
        normals = np.zeros_like(pts, dtype=np.float32)

    feats = [pts, normals.astype(np.float32)]

    # Neighborhood indices
    k = int(cfg.knn)
    try:
        dists, knn_idx = _knn_indices(pts, k)
    except Exception:
        # 最低限のフォールバック（追加特徴はゼロ）
        feats.append(np.zeros((N, 1), dtype=np.float32))  # normal_var
        feats.append(np.zeros((N, 1), dtype=np.float32))  # density
        feats.append(np.zeros((N, 5), dtype=np.float32))  # pca-derived
        return np.concatenate(feats, axis=1)

    # normal variance（角度の分散に類似）& density（近傍平均距離）
    nrm = normals.astype(np.float32)
    if cfg.add_normal_var:
        # cos(angle) variance with neighbors
        center_n = nrm
        neigh_n = nrm[knn_idx]
        # 正規化（ゼロ割防止）
        ln = np.linalg.norm(center_n, axis=1, keepdims=True) + 1e-6
        cn = (center_n / ln)[:, None, :]
        ln2 = np.linalg.norm(neigh_n, axis=2, keepdims=True) + 1e-6
        nn = neigh_n / ln2
        cosang = np.clip((cn * nn).sum(axis=2), -1.0, 1.0)
        nvar = cosang.var(axis=1)
        normal_var = nvar.astype(np.float32)[:, None]
    else:
        normal_var = np.zeros((N, 1), dtype=np.float32)

    if cfg.add_density:
        density = dists.mean(axis=1).astype(np.float32)[:, None]
    else:
        density = np.zeros((N, 1), dtype=np.float32)

    # PCA-derived measures（curvature proxies）— vectorized
    pca_feats = np.zeros((N, 5), dtype=np.float32)
    if cfg.add_curvature:
        # Batch covariance + eigenvalues (no Python loop)
        X = pts[knn_idx]  # (N, k, 3)
        X_centered = X - X.mean(axis=1, keepdims=True)  # (N, k, 3)
        C = np.einsum('nki,nkj->nij', X_centered, X_centered) / max(1, k - 1)  # (N, 3, 3)
        eigs = np.linalg.eigvalsh(C)  # (N, 3), sorted ascending
        eigs = eigs[:, ::-1].copy()  # l1 >= l2 >= l3
        eigs = np.maximum(eigs, 0.0)  # clamp negatives

        s = eigs.sum(axis=1, keepdims=True).clip(min=1e-9)  # (N, 1)
        e = eigs / s  # normalized eigenvalues
        e1, e2, e3 = e[:, 0], e[:, 1], e[:, 2]
        e1_safe = np.maximum(e1, 1e-9)

        linearity = (e1 - e2) / e1_safe
        planarity = (e2 - e3) / e1_safe
        sphericity = e3 / e1_safe
        anisotropy = (e1 - e3) / e1_safe
        # omnivariance: (l1*l2*l3)^(1/3), 0 where any eigenvalue <= 0
        valid = (eigs > 0).all(axis=1)
        omnivariance = np.zeros(N, dtype=np.float32)
        if valid.any():
            omnivariance[valid] = np.cbrt(eigs[valid].prod(axis=1)).astype(np.float32)

        lin, pla, sph, aniso, omni = _robust_standardize(
            [linearity.astype(np.float32), planarity.astype(np.float32),
             sphericity.astype(np.float32), anisotropy.astype(np.float32), omnivariance])
        pca_feats = np.stack([lin, pla, sph, aniso, omni], axis=1).astype(np.float32)

    # density/normal_varの簡易標準化
    normal_var_s, density_s = _robust_standardize([normal_var.squeeze(-1), density.squeeze(-1)])
    normal_var = normal_var_s[:, None]
    density = density_s[:, None]

    feats.extend([normal_var, density, pca_feats])
    return np.concatenate(feats, axis=1)
