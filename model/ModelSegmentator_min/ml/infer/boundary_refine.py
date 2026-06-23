from __future__ import annotations

import numpy as np


def bilateral_boundary_smooth(
    prob: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray | None = None,
    k: int = 16,
    iters: int = 3,
    alpha: float = 0.3,
    boundary_low: float = 0.2,
    boundary_high: float = 0.8,
) -> np.ndarray:
    """Bilateral boundary smoothing on tooth probabilities.

    Only smooths points near the prediction boundary (where neighbors
    disagree). Uses normal similarity to preserve sharp edges.

    Args:
        prob: (N,) tooth probability in [0,1]
        points: (N,3) xyz coordinates
        normals: (N,3) surface normals (optional, improves edge preservation)
        k: number of KNN neighbors
        iters: smoothing iterations
        alpha: smoothing strength (0=no change, 1=full replace)
        boundary_low: lower threshold for boundary candidate detection
        boundary_high: upper threshold for boundary candidate detection

    Returns:
        prob_refined: (N,) smoothed probability
    """
    N = points.shape[0]
    if N < 2:
        return prob.copy()

    kk = min(k + 1, N)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, indices = tree.query(points, k=kk)
    except Exception:
        # brute force fallback
        diff = points[:, None, :] - points[None, :, :]
        dist2 = (diff ** 2).sum(axis=2)
        indices = np.argpartition(dist2, kth=kk - 1, axis=1)[:, :kk]
        dists = np.take_along_axis(dist2, indices, axis=1) ** 0.5

    # exclude self (first column)
    indices = indices[:, 1:]
    dists = dists[:, 1:]

    # distance weights
    dist_w = 1.0 / (dists + 1e-6)  # (N, k)

    # normal similarity weights
    if normals is not None:
        normal_self = normals[:, None, :]           # (N, 1, 3)
        normal_neigh = normals[indices]              # (N, k, 3)
        normal_sim = np.sum(normal_self * normal_neigh, axis=-1)  # (N, k)
        normal_sim = np.clip(normal_sim, 0.0, 1.0)
    else:
        normal_sim = np.ones_like(dist_w)

    # combined weights (precomputed, constant across iterations)
    weight = dist_w * normal_sim  # (N, k)
    weight_sum = weight.sum(axis=1, keepdims=True) + 1e-8
    weight = weight / weight_sum  # (N, k) normalized

    p = prob.astype(np.float32).copy()

    for _ in range(iters):
        neigh_p = p[indices]  # (N, k)
        weighted_avg = (neigh_p * weight).sum(axis=1)  # (N,)

        # identify boundary candidates: points with mixed neighbor predictions
        neigh_mean = neigh_p.mean(axis=1)
        is_boundary = (neigh_mean > boundary_low) & (neigh_mean < boundary_high)

        # only update boundary candidates
        p[is_boundary] = (1.0 - alpha) * p[is_boundary] + alpha * weighted_avg[is_boundary]

    return p


def remove_small_components(
    mask: np.ndarray,
    points: np.ndarray,
    min_size: int = 20,
    k: int = 16,
) -> np.ndarray:
    """Remove small isolated predicted regions using KNN graph connectivity.

    Args:
        mask: (N,) boolean tooth mask
        points: (N,3) xyz coordinates
        min_size: minimum connected component size to keep
        k: KNN neighbors for graph construction

    Returns:
        mask_cleaned: (N,) boolean
    """
    tooth_idx = np.where(mask)[0]
    if len(tooth_idx) < min_size:
        return mask.copy()

    tooth_pts = points[tooth_idx]
    kk = min(k, len(tooth_idx))

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(tooth_pts)
        dists, indices = tree.query(tooth_pts, k=kk)
    except Exception:
        return mask.copy()

    # build sparse adjacency and find connected components
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(tooth_idx)
    adj = lil_matrix((n, n), dtype=np.bool_)
    for i in range(n):
        for j in range(kk):
            ni = indices[i, j]
            if ni != i:
                adj[i, ni] = True
                adj[ni, i] = True

    n_components, labels = connected_components(adj.tocsr(), directed=False)

    # count component sizes
    mask_out = mask.copy()
    for comp_id in range(n_components):
        comp_mask = labels == comp_id
        if comp_mask.sum() < min_size:
            # remove small component
            mask_out[tooth_idx[comp_mask]] = False

    return mask_out
