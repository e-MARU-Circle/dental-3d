from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

import numpy as np

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))


def _pca_axes(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    return V[:, order]  # columns are principal axes (pc1, pc2, pc3)


def _assign_quadrants(centroids: np.ndarray, jaw: str, flip_lr: bool) -> tuple[Dict[int, int], np.ndarray, np.ndarray]:
    axes = _pca_axes(centroids)
    pc1 = axes[:, 0]
    pc2 = axes[:, 1] if axes.shape[1] > 1 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x = centroids @ pc1
    right = x < np.median(x)
    left = ~right
    if flip_lr:
        right, left = left, right
    if jaw.lower().startswith('u'):
        quad_right = 1
        quad_left = 2
    else:
        quad_left = 3
        quad_right = 4
    mapping: Dict[int, int] = {}
    for i, is_right in enumerate(right):
        mapping[i] = quad_right if is_right else quad_left
    return mapping, pc1, pc2


def _order_within_quadrant(centroids: np.ndarray, idxs: list[int], pc1: np.ndarray, pc2: np.ndarray) -> list[int]:
    if not idxs:
        return []
    x_all = centroids @ pc1
    y_all = centroids @ pc2
    xm = np.median(x_all)
    ym = np.median(y_all)
    best_order: list[int] = []
    best_score: float | None = None
    for reverse in (False, True):
        order = sorted(idxs, key=lambda i: y_all[i], reverse=reverse)
        score = 0.0
        for rank, idx in enumerate(order):
            score += (rank + 1) * abs(float(y_all[idx] - ym))
        score += 0.1 * abs(float(x_all[order[0]] - xm))
        if best_score is None or score < best_score:
            best_score = score
            best_order = order
    return best_order


def main() -> None:
    ap = argparse.ArgumentParser(description='Assign FDI numbers to instance clusters (baseline)')
    ap.add_argument('--points', type=str, required=True, help='path to .npz with points (original sample)')
    ap.add_argument('--pred', type=str, required=True, help='path to _pred.npz with inst labels (from segment_and_instance)')
    ap.add_argument('--out', type=str, required=True, help='output path (.npz)')
    ap.add_argument('--jaw', type=str, default='upper', choices=['upper', 'lower'])
    ap.add_argument('--flip-lr', action='store_true', help='flip left/right if orientation is reversed')
    args = ap.parse_args()

    pts = np.load(args.points)['points'].astype(np.float32)
    inst = np.load(args.pred)['inst'].astype(np.int32)

    tooth_sel = inst >= 0
    if not np.any(tooth_sel):
        np.savez_compressed(args.out, fdi=np.zeros((pts.shape[0],), dtype=np.int32))
        print('[WARN] no tooth instances; wrote zeros')
        return

    # compute instance centroids
    ids = sorted([i for i in np.unique(inst) if i >= 0])
    centroids = np.stack([pts[inst == i].mean(axis=0) for i in ids], axis=0)

    axes = _pca_axes(centroids)
    pc1 = axes[:, 0]
    quad_map, pc1, pc2 = _assign_quadrants(centroids, args.jaw, args.flip_lr)

    # Group indices by quadrant
    quad_indices: Dict[int, list[int]] = {1: [], 2: [], 3: [], 4: []}
    for k, q in quad_map.items():
        quad_indices[q].append(k)

    # FDI sequences by quadrant
    if args.jaw.lower().startswith('u'):
        seq = {1: [11,12,13,14,15,16,17,18], 2: [21,22,23,24,25,26,27,28], 3: [], 4: []}
    else:
        seq = {3: [31,32,33,34,35,36,37,38], 4: [41,42,43,44,45,46,47,48], 1: [], 2: []}

    # Assign FDI to instances
    inst_to_fdi: Dict[int, int] = {i: 0 for i in ids}
    for q, idxs in quad_indices.items():
        if not idxs or not seq.get(q):
            continue
        ordered = _order_within_quadrant(centroids, idxs, pc1, pc2)
        for i, iid in enumerate(ordered[:len(seq[q])]):
            inst_to_fdi[ids[iid]] = seq[q][i]

    # Map to per-point label
    fdi = np.zeros((pts.shape[0],), dtype=np.int32)
    for iid, label in inst_to_fdi.items():
        if label <= 0:
            continue
        fdi[inst == iid] = label

    np.savez_compressed(args.out, fdi=fdi)
    print(f'[OK] wrote {args.out}')


if __name__ == '__main__':
    main()
