"""3D Gaussian heatmap generation for landmarks (skeleton)."""
from __future__ import annotations

import numpy as np


def gaussian_heatmap(points: np.ndarray, lm_xyz: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Compute per-point heatmap values given landmark xyz.

    points: (N,3), lm_xyz: (M,3) for M landmarks.
    returns: (N,M) heat values in [0,1].
    """
    if lm_xyz.size == 0:
        return np.zeros((points.shape[0], 0), dtype=np.float32)
    dif = points[:, None, :] - lm_xyz[None, :, :]
    d2 = (dif ** 2).sum(axis=2)
    h = np.exp(-0.5 * d2 / (sigma ** 2))
    return h.astype(np.float32)

