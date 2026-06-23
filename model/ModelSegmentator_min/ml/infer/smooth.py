from __future__ import annotations

import numpy as np


def knn_laplacian_smooth(prob: np.ndarray, points: np.ndarray, k: int = 16, iters: int = 5, lam: float = 0.5) -> np.ndarray:
    """Simple Laplacian smoothing on probability over kNN graph.

    prob: (N,) in [0,1]
    points: (N,3)
    Returns smoothed probability.
    """
    try:
        from scipy.spatial import cKDTree as KDTree  # type: ignore
        tree = KDTree(points)
        d, idx = tree.query(points, k=min(k, points.shape[0]))
    except Exception:
        # brute force
        N = points.shape[0]
        kk = min(k, N)
        dist2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        idx = np.argpartition(dist2, kth=kk-1, axis=1)[:, :kk]
        d = np.take_along_axis(dist2, idx, axis=1) ** 0.5

    p = prob.astype(np.float32)
    for _ in range(int(iters)):
        neigh_p = p[idx]
        w = 1.0 / (d + 1e-6)
        w = w / (w.sum(axis=1, keepdims=True) + 1e-6)
        avg = (neigh_p * w).sum(axis=1)
        p = (1.0 - lam) * p + lam * avg
    return p

