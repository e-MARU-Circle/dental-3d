from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

# ensure repo on path
_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from ml.models.backbone_pointnet2 import DummyBackbone
from ml.models.pointnet2_backbone import PointNet2Backbone, PointNet2BackboneLite
from ml.models.heads import HeadSem2, HeadEmb, HeadOffset
from ml.data.preprocess_features import compute_features, FeatureConfig
from ml.infer.assign_fdi import _pca_axes
from ml.infer.cluster_stage2 import try_dbscan
from ml.infer.clustering import spatial_meanshift_cluster, split_large_spatial, merge_nearby_clusters
from ml.utils.exclude import read_exclude_list

# TTA rotation angles (8 rotations for best results)
TTA_ANGLES_8 = [0, 45, 90, 135, 180, 225, 270, 315]
TTA_ANGLES_4 = [0, 90, 180, 270]


def _device() -> torch.device:
    return torch.device(
        'cuda' if torch.cuda.is_available() else (
            'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'
        )
    )


def _load_stage1(in_channels: int, ckpt: Path, device: torch.device, state: dict | None = None, backbone_type: str = "pointnet2_lite") -> tuple[torch.nn.Module, HeadSem2]:
    """Load Stage-1 model with specified backbone type."""
    if backbone_type == "pointnet2_lite":
        bb1 = PointNet2BackboneLite(in_channels=in_channels, out_dim=128).to(device)
    else:
        bb1 = DummyBackbone(in_channels=in_channels, out_dim=128).to(device)
    head1 = HeadSem2(in_dim=128).to(device)
    if state is None:
        state = torch.load(ckpt, map_location=device)
    bb1.load_state_dict(state.get('backbone', {}), strict=False)
    head1.load_state_dict(state.get('head_sem2', {}), strict=False)
    bb1.eval(); head1.eval()
    return bb1, head1


def _rotate_z(xyz: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate points around Z-axis."""
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return xyz @ rot.T


def _load_stage2(in_channels: int, emb_dim: int, ckpt: Path, device: torch.device) -> tuple[DummyBackbone, HeadEmb]:
    state = torch.load(ckpt, map_location=device)
    state_head = state.get('head_emb', {})
    # prefer checkpoint embedding dimension if available
    if 'mlp.2.weight' in state_head:
        emb_dim = int(state_head['mlp.2.weight'].shape[0])
    bb2 = DummyBackbone(in_channels=in_channels, out_dim=128).to(device)
    head2 = HeadEmb(in_dim=128, out_dim=emb_dim).to(device)
    bb2.load_state_dict(state.get('backbone', {}), strict=False)
    head2.load_state_dict(state_head, strict=False)
    bb2.eval(); head2.eval()
    return bb2, head2


def _infer_sem2(feats: np.ndarray, bb: torch.nn.Module, head: HeadSem2, device: torch.device, thr: float = 0.5, tta_angles: list | None = None, refine: bool = False, refine_k: int = 16, refine_iters: int = 3, refine_alpha: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """
    Infer Stage-1 segmentation with optional TTA.

    Args:
        feats: (N, C) input features, first 3 are xyz, 3:6 are normals
        bb: backbone model
        head: head model
        device: torch device
        thr: classification threshold (default: 0.5, optimized: 0.90)
        tta_angles: list of rotation angles for TTA. None for no TTA.

    Returns:
        prob: (N,) tooth probability
        mask: (N,) boolean tooth mask
    """
    if tta_angles is None or len(tta_angles) <= 1:
        # No TTA
        x = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            f = bb(x)
            logits = head(f)
            prob = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()
    else:
        # TTA with multiple rotations
        xyz = feats[:, :3]
        normals = feats[:, 3:6] if feats.shape[1] >= 6 else None
        other = feats[:, 6:] if feats.shape[1] > 6 else None

        probs_sum = np.zeros(len(feats), dtype=np.float32)

        with torch.no_grad():
            for angle in tta_angles:
                xyz_rot = _rotate_z(xyz, angle)
                if normals is not None:
                    normals_rot = _rotate_z(normals, angle)
                    if other is not None:
                        feats_rot = np.concatenate([xyz_rot, normals_rot, other], axis=1)
                    else:
                        feats_rot = np.concatenate([xyz_rot, normals_rot], axis=1)
                else:
                    feats_rot = xyz_rot if other is None else np.concatenate([xyz_rot, other], axis=1)

                x = torch.from_numpy(feats_rot.astype(np.float32)).unsqueeze(0).to(device)
                f = bb(x)
                logits = head(f)
                probs_sum += torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()

        prob = probs_sum / len(tta_angles)

    # Bilateral boundary refinement (post-processing)
    if refine:
        from ml.infer.boundary_refine import bilateral_boundary_smooth
        xyz = feats[:, :3]
        normals = feats[:, 3:6] if feats.shape[1] >= 6 else None
        prob = bilateral_boundary_smooth(prob, xyz, normals,
                                         k=refine_k, iters=refine_iters, alpha=refine_alpha)

    mask = (prob > float(thr)).astype(np.bool_)
    return prob, mask


def _infer_emb(feats: np.ndarray, sel: np.ndarray, bb: DummyBackbone, head: HeadEmb, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(feats[sel].astype(np.float32)).to(device)
    with torch.no_grad():
        f = bb(x)
        emb = head(f).detach().cpu().numpy()
    # standardize for clustering stability
    mu = emb.mean(axis=0, keepdims=True)
    std = emb.std(axis=0, keepdims=True) + 1e-6
    return (emb - mu) / std


def _load_stage2_offset(in_channels: int, ckpt: Path, device: torch.device) -> tuple[torch.nn.Module, HeadOffset]:
    """Load Stage-2 model for offset-based instance segmentation."""
    state = torch.load(ckpt, map_location=device)
    cfg = state.get('config', {}) or {}
    backbone_type = str(cfg.get('model', {}).get('backbone', 'pointnet2_lite')).lower()
    out_dim = int(cfg.get('model', {}).get('feature_dim', 128))
    center_cfg = cfg.get('model', {}).get('center_offset', {}) or {}
    hidden = int(center_cfg.get('hidden', 256))

    if backbone_type in ('pointnet2', 'pointnet2_full'):
        bb = PointNet2Backbone(in_channels=in_channels, out_dim=out_dim, use_msg=False).to(device)
    elif backbone_type == 'pointnet2_lite':
        bb = PointNet2BackboneLite(in_channels=in_channels, out_dim=out_dim).to(device)
    else:
        bb = DummyBackbone(in_channels=in_channels, out_dim=out_dim).to(device)

    head_off = HeadOffset(in_dim=out_dim, hidden=hidden, out_dim=3).to(device)
    bb.load_state_dict(state.get('backbone', {}), strict=False)
    head_off.load_state_dict(state.get('head_offset', {}), strict=False)
    bb.eval(); head_off.eval()
    return bb, head_off


def _infer_offset(feats: np.ndarray, sel: np.ndarray, bb: torch.nn.Module, head_off: HeadOffset, device: torch.device, max_points: int = 0) -> np.ndarray:
    """Predict per-point 3D offset to instance centroid.

    If max_points > 0 and tooth points exceed max_points, subsample for
    inference then interpolate offsets back to all points via KDTree.
    """
    feats_tooth = feats[sel].astype(np.float32)
    N = feats_tooth.shape[0]

    if max_points > 0 and N > max_points:
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(N, size=max_points, replace=False)
        x = torch.from_numpy(feats_tooth[idx_sub]).unsqueeze(0).to(device)
        with torch.no_grad():
            f = bb(x)
            offsets_sub = head_off(f).squeeze(0).cpu().numpy()  # (max_points, 3)
        # Interpolate back to all tooth points using KDTree on XYZ (first 3 cols)
        from sklearn.neighbors import KDTree
        tree = KDTree(feats_tooth[idx_sub, :3])
        _, nn_idx = tree.query(feats_tooth[:, :3], k=1)
        offsets = offsets_sub[nn_idx.flatten()]
        print(f'  [SUBSAMPLE] offset inference: {N} -> {max_points} pts, interpolated back')
        return offsets

    x = torch.from_numpy(feats_tooth).unsqueeze(0).to(device)
    with torch.no_grad():
        f = bb(x)
        offsets = head_off(f).squeeze(0).cpu().numpy()
    return offsets


def _normalize_points_np(points: np.ndarray) -> np.ndarray:
    """Normalize points to [-1, 1] range (matches training normalization)."""
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    rng = np.maximum(mx - mn, 1e-6)
    return 2.0 * (points - mn) / rng - 1.0


def _subsample_points(points: np.ndarray, inst: np.ndarray, max_points: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, inst
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx], inst[idx]


def _normalize_points(points: np.ndarray, jaw: str, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = points.astype(np.float32, copy=True)
    center_cfg = cfg.get('center', {}) or {}
    if center_cfg.get('enable', False) and pts.shape[0] > 0:
        centroid = pts.mean(axis=0, keepdims=True)
        pts = pts - centroid
    axes_cfg = cfg.get('pca_align', {}) or {}
    pc1 = pc2 = None
    if axes_cfg.get('enable', False) and pts.shape[0] >= 8:
        try:
            axes = _pca_axes(pts)
            pc1 = axes[:, 0]
            pc2 = axes[:, 1] if axes.shape[1] > 1 else None
            pts = pts @ axes
        except Exception:
            pass
    scale_cfg = cfg.get('scale', {}) or {}
    if scale_cfg.get('enable', False):
        method = str(scale_cfg.get('method', 'std')).lower()
        if method == 'std':
            std = pts.std(axis=0, keepdims=True) + 1e-6
            pts = pts / std
        axis_weights = scale_cfg.get('axis_weights')
        if axis_weights:
            axis_weights = np.asarray(axis_weights, dtype=np.float32)
            if axis_weights.size == pts.shape[1]:
                pts = pts * axis_weights
    return pts, pc1, pc2


def _refine_large_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    base_eps: float,
    base_min_samples: int,
    split_cfg: dict,
) -> np.ndarray:
    size_threshold = int(split_cfg.get('size_threshold', 8000) or 0)
    if size_threshold <= 0:
        return labels
    eps_conf = split_cfg.get('eps_scale', 0.5)
    if isinstance(eps_conf, (list, tuple)):
        scales = [float(s) for s in eps_conf if float(s) > 0]
    else:
        try:
            scales = [float(eps_conf)]
        except Exception:
            scales = [0.5]
    scales = [s for s in scales if s > 0]
    if not scales:
        scales = [0.5]
    min_conf = split_cfg.get('min_samples', max(10, base_min_samples // 2))
    if isinstance(min_conf, (list, tuple)):
        mins = [int(v) for v in min_conf]
    else:
        mins = [int(min_conf)]
    mins = [m if m > 0 else max(5, base_min_samples // 2) for m in mins]
    max_iter = int(split_cfg.get('max_iter', 3) or 3)
    kmeans_cfg = split_cfg.get('kmeans', {}) or {}
    enable_kmeans = bool(kmeans_cfg.get('enable', False))
    min_kmeans_size = int(kmeans_cfg.get('min_size', size_threshold))
    min_ratio = float(kmeans_cfg.get('min_ratio', 0.1))
    min_sep = float(kmeans_cfg.get('min_sep', 1.0))
    max_kmeans_iter = int(kmeans_cfg.get('max_iter', 20) or 20)

    updated = labels.copy()
    next_id = int(updated[updated >= 0].max()) + 1 if np.any(updated >= 0) else 0
    for _ in range(max_iter):
        changed = False
        unique_ids = sorted(int(c) for c in np.unique(updated) if c >= 0)
        for cluster_id in unique_ids:
            idx = np.where(updated == cluster_id)[0]
            if idx.size <= size_threshold:
                continue
            sub_labels = None
            for scale in scales:
                for sub_min in mins:
                    trial = try_dbscan(features[idx], eps=max(1e-3, base_eps * scale), min_samples=sub_min)
                    unique_sub = sorted(int(s) for s in np.unique(trial) if s >= 0)
                    if len(unique_sub) > 1:
                        sub_labels = trial
                        break
                if sub_labels is not None:
                    break
            if sub_labels is None and enable_kmeans and idx.size >= max(min_kmeans_size, 4):
                km_labels = _kmeans_bisect(features[idx], max_iter=max_kmeans_iter)
                if km_labels is not None:
                    count0 = (km_labels == 0).sum()
                    count1 = (km_labels == 1).sum()
                    if min(count0, count1) / idx.size >= min_ratio:
                        sep = _cluster_separation(features[idx], km_labels == 0, km_labels == 1)
                        if sep >= min_sep:
                            sub_labels = km_labels
            if sub_labels is None:
                continue
            unique_sub = sorted(int(s) for s in np.unique(sub_labels) if s >= 0)
            if len(unique_sub) <= 1:
                continue
            changed = True
            updated[idx] = -1
            for sub_id in unique_sub:
                sel = idx[sub_labels == sub_id]
                if sel.size == 0:
                    continue
                updated[sel] = next_id
                next_id += 1
        if not changed:
            break
    return updated


def _kmeans_bisect(points: np.ndarray, max_iter: int = 20) -> np.ndarray | None:
    if points.shape[0] < 2:
        return None
    rng = np.random.default_rng(0)
    idx0 = rng.integers(0, points.shape[0])
    diff = points - points[idx0]
    idx1 = np.argmax(np.linalg.norm(diff, axis=1))
    c0 = points[idx0]
    c1 = points[idx1]
    labels = np.zeros(points.shape[0], dtype=np.int32)
    for _ in range(max_iter):
        dist0 = np.linalg.norm(points - c0, axis=1)
        dist1 = np.linalg.norm(points - c1, axis=1)
        new_labels = (dist1 < dist0).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
            return None
        c0 = points[labels == 0].mean(axis=0)
        c1 = points[labels == 1].mean(axis=0)
    if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
        return None
    return labels


def _cluster_separation(points: np.ndarray, mask0: np.ndarray, mask1: np.ndarray) -> float:
    centroid0 = points[mask0].mean(axis=0)
    centroid1 = points[mask1].mean(axis=0)
    sep = np.linalg.norm(centroid0 - centroid1)
    spread0 = np.sqrt(((points[mask0] - centroid0) ** 2).sum(axis=1).mean()) + 1e-6
    spread1 = np.sqrt(((points[mask1] - centroid1) ** 2).sum(axis=1).mean()) + 1e-6
    return sep / (spread0 + spread1)


def _cluster_stats(points: np.ndarray, labels: np.ndarray) -> dict | None:
    ids = [int(i) for i in np.unique(labels) if i >= 0]
    if not ids:
        return None
    sizes = np.array([(labels == cid).sum() for cid in ids], dtype=np.int64)
    centroids = np.stack([points[labels == cid].mean(axis=0) for cid in ids], axis=0)
    nearest = None
    if centroids.shape[0] > 1:
        diff = centroids[:, None, :] - centroids[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        flat_idx = np.argmin(dist)
        i, j = divmod(int(flat_idx), dist.shape[1])
        nearest = {
            'pair': (ids[i], ids[j]),
            'distance': float(dist[i, j]),
        }
    return {
        'count': len(ids),
        'sizes': {
            'min': int(sizes.min()),
            'median': float(np.median(sizes)),
            'max': int(sizes.max()),
        },
        'nearest': nearest,
    }


def _merge_small_clusters(points: np.ndarray, labels: np.ndarray, cfg: dict) -> np.ndarray:
    if not cfg.get('enable', False):
        return labels
    thresh = float(cfg.get('distance', 3.0) or 3.0)
    if thresh <= 0:
        return labels
    min_points = int(cfg.get('min_points', 0) or 0)
    unique_ids = [int(i) for i in np.unique(labels) if i >= 0]
    if len(unique_ids) <= 1:
        return labels
    centroids = {}
    counts = {}
    for cid in unique_ids:
        mask = labels == cid
        pts = points[mask]
        if pts.size == 0:
            continue
        centroids[cid] = pts.mean(axis=0)
        counts[cid] = pts.shape[0]
    merged = {cid: cid for cid in unique_ids}

    def find(x):
        while merged[x] != x:
            merged[x] = merged[merged[x]]
            x = merged[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            merged[rb] = ra

    for i in range(len(unique_ids)):
        for j in range(i + 1, len(unique_ids)):
            cid_i = unique_ids[i]
            cid_j = unique_ids[j]
            # optional guard: only consider merging when at least one cluster is small
            if min_points > 0:
                if counts.get(cid_i, 0) > min_points and counts.get(cid_j, 0) > min_points:
                    continue
            if np.linalg.norm(centroids[cid_i] - centroids[cid_j]) < thresh:
                union(cid_i, cid_j)

    new_labels = labels.copy()
    mapping = {}
    next_id = 0
    for cid in unique_ids:
        root = find(cid)
        if root not in mapping:
            mapping[root] = next_id
            next_id += 1
        new_labels[labels == cid] = mapping[root]
    return new_labels


def main() -> None:
    ap = argparse.ArgumentParser(description='Stage-1/2 inference: sem2 -> embedding -> DBSCAN clustering')
    ap.add_argument('--config', type=str, default=str(_repo / 'configs/stage2.yaml'))
    ap.add_argument('--input', type=str, nargs='+', default=[str(_repo / 'data/sample_npz/upper')],
                    help='One or more input directories (e.g. .../upper .../lower)')
    ap.add_argument('--ckpt1', type=str, default=str(_repo / 'ckpts/stage1_last.pth'))
    ap.add_argument('--ckpt2', type=str, default=str(_repo / 'ckpts/stage2_last.pth'))
    ap.add_argument('--out', type=str, default=str(_repo / 'pred/seg_inst'))
    ap.add_argument('--thr', type=float, default=0.90, help='tooth probability threshold (optimized default: 0.90)')
    ap.add_argument('--tta', type=int, default=8, help='Number of TTA rotations (0=disabled, 4 or 8 recommended)')
    ap.add_argument('--backbone', type=str, default='pointnet2_lite', choices=['pointnet2_lite', 'dummy'], help='Stage-1 backbone type')
    ap.add_argument('--exclude-list', type=str, default=str(_repo / 'configs/exclude_cases.txt'), help='path to newline-delimited list of case stems to skip')
    ap.add_argument('--cluster-method', type=str, default='offset_voting',
                    choices=['spatial_meanshift', 'dbscan', 'hdbscan', 'offset_voting'],
                    help='Clustering method: offset_voting (best, Dice=0.99), spatial_meanshift, dbscan, hdbscan')
    ap.add_argument('--offset-eps', type=float, default=0.03, help='DBSCAN eps for offset_voting (default: 0.03)')
    ap.add_argument('--offset-min-samples', type=int, default=100, help='DBSCAN min_samples for offset_voting (default: 100)')
    ap.add_argument('--dbscan-subsample', type=int, default=10000, help='Subsample N points for DBSCAN then propagate (0=disabled, default: 10000)')
    ap.add_argument('--offset-max-points', type=int, default=40000, help='Max tooth points for offset inference; larger files are subsampled+interpolated (0=disabled, default: 40000)')
    ap.add_argument('--merge-threshold', type=float, default=0.05, help='Merge clusters with shifted centroids closer than this (0=disabled, default: 0.05)')
    ap.add_argument('--refine', action='store_true', help='Apply bilateral boundary refinement on Stage-1 probabilities')
    ap.add_argument('--refine-k', type=int, default=16, help='KNN neighbors for boundary refinement (default: 16)')
    ap.add_argument('--refine-iters', type=int, default=3, help='Smoothing iterations for boundary refinement (default: 3)')
    ap.add_argument('--refine-alpha', type=float, default=0.3, help='Smoothing strength for boundary refinement (default: 0.3)')
    ap.add_argument('--bandwidth', type=float, default=4.0,
                    help='MeanShift bandwidth for spatial_meanshift (default: 4.0, optimal for dental)')
    ap.add_argument('--landmarks', action='store_true', help='Enable landmark prediction for each tooth instance')
    ap.add_argument('--landmark-ckpt', type=str, default=str(_repo / 'ckpts/landmark_vote_v2_best.pth'),
                    help='Landmark model checkpoint (v1 or v2 auto-detected)')
    ap.add_argument('--landmark-max-points', type=int, default=2048,
                    help='Max points per tooth for landmark inference (default: 2048)')
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    in_dirs = [Path(d) for d in args.input]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for in_dir in in_dirs:
        files.extend(p for p in sorted(in_dir.glob('*.npz')) if not p.name.endswith('_feats.npz'))
    exclude_cases = read_exclude_list(args.exclude_list)
    if exclude_cases:
        files = [p for p in files if p.stem not in exclude_cases]
    if not files:
        print(f'[WARN] no npz under {in_dirs}')
        return

    ckpt1 = Path(args.ckpt1)
    state1 = None
    feat_cfg = FeatureConfig()
    try:
        state1 = torch.load(ckpt1, map_location="cpu")
        feat_raw = (state1.get("config", {}) or {}).get("model", {}).get("feat", {}) or {}
        feat_cfg = FeatureConfig(**{k: feat_raw[k] for k in feat_raw if k in FeatureConfig.__annotations__})
    except Exception:
        state1 = None

    # Infer feature dimension
    sample = np.load(files[0])
    feats0 = sample.get('feats')
    if feats0 is None:
        feats0 = compute_features(sample['points'].astype(np.float32), sample.get('normals'), feat_cfg)
    in_channels = int(feats0.shape[-1])
    emb_dim = int(cfg['model'].get('emb_dim', 128))

    device = _device()
    bb1, head1 = _load_stage1(in_channels, ckpt1, device, state=state1, backbone_type=args.backbone)

    # Load Stage-2 model based on cluster method
    bb2 = head2 = bb2_off = head2_off = None
    if args.cluster_method == 'offset_voting':
        bb2_off, head2_off = _load_stage2_offset(in_channels, Path(args.ckpt2), device)
        print(f'[INFO] Loaded offset model from {args.ckpt2}')
    else:
        bb2, head2 = _load_stage2(in_channels, emb_dim, Path(args.ckpt2), device)

    # Setup TTA angles
    if args.tta <= 0:
        tta_angles = None
    elif args.tta == 4:
        tta_angles = TTA_ANGLES_4
    elif args.tta == 8:
        tta_angles = TTA_ANGLES_8
    else:
        tta_angles = [360.0 * i / args.tta for i in range(args.tta)]

    print(f"[INFO] Stage-1 backbone: {args.backbone}")
    print(f"[INFO] TTA: {args.tta} rotations" if tta_angles else "[INFO] TTA: disabled")
    print(f"[INFO] Threshold: {args.thr}")
    cluster_info = ""
    if args.cluster_method == 'spatial_meanshift':
        cluster_info = f" (bandwidth={args.bandwidth})"
    elif args.cluster_method == 'offset_voting':
        cluster_info = f" (eps={args.offset_eps}, min_samples={args.offset_min_samples}, max_pts={args.offset_max_points})"
    print(f"[INFO] Clustering: {args.cluster_method}{cluster_info}")
    print(f"[INFO] Device: {device}")

    # Load landmark model if enabled
    lm_backbone = lm_head = None
    if args.landmarks:
        from ml.infer.landmark_decode import load_landmark_model
        lm_ckpt = Path(args.landmark_ckpt)
        if lm_ckpt.exists():
            lm_backbone, lm_head = load_landmark_model(lm_ckpt, device)
        else:
            print(f"[WARN] Landmark checkpoint not found: {lm_ckpt}, landmarks disabled")
            args.landmarks = False

    cluster_cfg = cfg.get('cluster', {}) or {}
    eps = float(cluster_cfg.get('eps', 0.7))
    min_samples = int(cluster_cfg.get('min_samples', 300))
    split_cfg = cluster_cfg.get('split_large', {}) or {}
    split_enabled = bool(split_cfg.get('enable', False))
    log_stats = bool(cluster_cfg.get('log_stats', False))

    # --- Resume-aware main loop ---
    total = len(files)
    skipped = 0
    done = 0
    errors = []
    t_start = time.time()

    # Count already-completed files for accurate progress
    for p in files:
        if (out_dir / (p.stem + '_pred.npz')).exists():
            skipped += 1

    remaining = total - skipped
    if skipped > 0:
        print(f'[RESUME] {skipped}/{total} already done, {remaining} remaining')

    processed_idx = 0  # counts only newly processed files (for ETA)

    for file_idx, p in enumerate(files):
        out = out_dir / (p.stem + '_pred.npz')
        if out.exists():
            continue

        processed_idx += 1
        progress = skipped + done + 1
        # ETA calculation
        if done > 0:
            elapsed = time.time() - t_start
            avg_sec = elapsed / done
            eta_sec = avg_sec * (remaining - done)
            eta_str = f"ETA {eta_sec / 60:.0f}m" if eta_sec > 60 else f"ETA {eta_sec:.0f}s"
        else:
            eta_str = "ETA ..."
        print(f'[{progress}/{total}] {p.stem}  ({eta_str})')

        try:
            d = np.load(p)
            pts = d['points'].astype(np.float32)
            nrm = d.get('normals')
            feats = d.get('feats')
            if feats is None:
                feats = compute_features(pts, nrm, feat_cfg)

            prob, sel = _infer_sem2(feats, bb1, head1, device, thr=float(args.thr), tta_angles=tta_angles,
                                    refine=args.refine, refine_k=args.refine_k,
                                    refine_iters=args.refine_iters, refine_alpha=args.refine_alpha)
            if not np.any(sel):
                sel = np.ones((pts.shape[0],), dtype=bool)

            subsample_cfg = cluster_cfg.get('subsample', {}) or {}
            max_cluster_points = int(subsample_cfg.get('max_points', 0) or 0)
            subsample_seed = int(subsample_cfg.get('seed', 42) or 42)
            norm_cfg = cluster_cfg.get('normalize_points', {}) or {}

            feats_sel = feats[sel]
            points_sel_full = pts[sel]
            points_norm_full, _, _ = _normalize_points(points_sel_full, p.stem.split('_')[-1], norm_cfg)
            points_sel = points_sel_full
            subsample_labels = None
            if max_cluster_points > 0 and feats_sel.shape[0] > max_cluster_points:
                feats_sel, subsample_labels = _subsample_points(feats_sel, np.arange(feats_sel.shape[0]), max_cluster_points, seed=subsample_seed)
                points_sel = points_sel[subsample_labels]

            points_norm = points_norm_full if subsample_labels is None else points_norm_full[subsample_labels]

            # Clustering based on selected method
            if args.cluster_method == 'offset_voting':
                # Offset-based instance segmentation (Test Dice=0.99)
                offsets_pred = _infer_offset(feats, sel, bb2_off, head2_off, device, max_points=args.offset_max_points)
                pts_norm_off = _normalize_points_np(points_sel_full)
                shifted = pts_norm_off + offsets_pred
                # Subsample DBSCAN for speed (50k→10k: 1.4x faster, same Dice)
                sub_n = args.dbscan_subsample
                N = shifted.shape[0]
                if sub_n > 0 and N > sub_n:
                    rng = np.random.default_rng(42)
                    idx_sub = rng.choice(N, size=sub_n, replace=False)
                    ms_adj = max(5, int(args.offset_min_samples * sub_n / N))
                    sub_labels = try_dbscan(shifted[idx_sub], eps=args.offset_eps, min_samples=ms_adj)
                    valid = (sub_labels >= 0)
                    if valid.sum() > 0:
                        from sklearn.neighbors import KDTree
                        tree = KDTree(shifted[idx_sub[valid]])
                        _, nn_idx = tree.query(shifted, k=1)
                        lab_sel = sub_labels[valid][nn_idx.flatten()].copy()
                    else:
                        lab_sel = -np.ones(N, dtype=np.int32)
                else:
                    lab_sel = try_dbscan(shifted, eps=args.offset_eps, min_samples=args.offset_min_samples)
                # Assign noise points to nearest cluster
                noise_mask = (lab_sel == -1)
                if noise_mask.sum() > 0 and (~noise_mask).sum() > 0:
                    from sklearn.neighbors import KDTree
                    tree = KDTree(shifted[~noise_mask])
                    _, nn_idx = tree.query(shifted[noise_mask], k=1)
                    lab_sel[noise_mask] = lab_sel[~noise_mask][nn_idx.flatten()]
                # Merge close clusters in shifted space (fixes over-segmentation)
                merge_thr = args.merge_threshold
                if merge_thr > 0:
                    unique_ids = [int(i) for i in np.unique(lab_sel) if i >= 0]
                    if len(unique_ids) > 1:
                        centroids = np.array([shifted[lab_sel == cid].mean(axis=0) for cid in unique_ids])
                        parent = {cid: cid for cid in unique_ids}
                        def _find(x):
                            while parent[x] != x:
                                parent[x] = parent[parent[x]]
                                x = parent[x]
                            return x
                        for i in range(len(unique_ids)):
                            for j in range(i + 1, len(unique_ids)):
                                if np.linalg.norm(centroids[i] - centroids[j]) < merge_thr:
                                    ri, rj = _find(unique_ids[i]), _find(unique_ids[j])
                                    if ri != rj:
                                        parent[rj] = ri
                        new_labels = lab_sel.copy()
                        mapping = {}
                        next_id = 0
                        for cid in unique_ids:
                            root = _find(cid)
                            if root not in mapping:
                                mapping[root] = next_id
                                next_id += 1
                            new_labels[lab_sel == cid] = mapping[root]
                        if next_id < len(unique_ids):
                            print(f'  [MERGE] {len(unique_ids)} -> {next_id} clusters (thr={merge_thr})')
                        lab_sel = new_labels
                features_for_split = shifted
            elif args.cluster_method == 'spatial_meanshift':
                # Spatial MeanShift + per-jaw post-processing (Dice=0.66-0.67)
                jaw_type = 'upper' if 'upper' in p.stem.lower() else ('lower' if 'lower' in p.stem.lower() else None)
                lab_sel = spatial_meanshift_cluster(points_sel_full, bandwidth=args.bandwidth, jaw=jaw_type)
                if jaw_type and 'lower' in jaw_type:
                    lab_sel = split_large_spatial(points_sel_full, lab_sel, size_threshold=4500, sub_bandwidth=3.4)
                    lab_sel = merge_nearby_clusters(points_sel_full, lab_sel, min_size=2000, max_merge_dist=5.0)
                elif jaw_type and 'upper' in jaw_type:
                    lab_sel = merge_nearby_clusters(points_sel_full, lab_sel, min_size=3500, max_merge_dist=5.0)
                features_for_split = points_norm_full
            else:
                # Embedding-based methods (legacy)
                z = _infer_emb(feats, sel, bb2, head2, device)
                features_for_split = z
                if subsample_labels is not None:
                    z_sub = np.concatenate([z[subsample_labels], points_norm], axis=1)
                    lab_sub = try_dbscan(z_sub, eps=eps, min_samples=min_samples)
                    lab_sel = -np.ones(z.shape[0], dtype=np.int32)
                    lab_sel[subsample_labels] = lab_sub
                else:
                    z_cat = np.concatenate([z, points_norm], axis=1)
                    lab_sel = try_dbscan(z_cat, eps=eps, min_samples=min_samples)
                    features_for_split = z_cat
            # determine jaw-specific split config
            cluster_cfg_use = split_cfg
            jaw_key = p.stem.split('_')[-1].lower()
            jaw_cfg = split_cfg.get('jaws', {}) if isinstance(split_cfg, dict) else {}
            if isinstance(jaw_cfg, dict) and jaw_key in jaw_cfg:
                cluster_cfg_use = {**split_cfg, **jaw_cfg[jaw_key]}
            if split_enabled:
                lab_sel = _refine_large_clusters(features_for_split, lab_sel, eps, min_samples, cluster_cfg_use)
            merge_cfg = cluster_cfg.get('merge_small', {}) or {}
            if merge_cfg.get('enable', False):
                merge_cfg_use = merge_cfg
                jaw_merge_cfg = merge_cfg.get('jaws', {})
                if isinstance(jaw_merge_cfg, dict) and jaw_key in jaw_merge_cfg:
                    merge_cfg_use = {**merge_cfg, **jaw_merge_cfg[jaw_key]}
                lab_sel = _merge_small_clusters(points_sel_full, lab_sel, merge_cfg_use)
            inst_all = -np.ones((pts.shape[0],), dtype=np.int32)
            inst_all[sel] = lab_sel

            stats = None
            if log_stats:
                stats = _cluster_stats(points_sel_full, lab_sel)
                if stats:
                    nearest = stats.get('nearest')
                    nearest_str = 'n/a'
                    if nearest:
                        pair = nearest.get('pair', ('?', '?'))
                        dist = nearest.get('distance', float('nan'))
                        nearest_str = f"{pair[0]},{pair[1]}:{dist:.2f}"
                    size_info = stats['sizes']
                    print(
                        f"[STATS] {p.stem}: clusters={stats['count']} "
                        f"size[min/med/max]={size_info['min']}/{size_info['median']:.1f}/{size_info['max']} "
                        f"nearest_pair={nearest_str}"
                    )
                    stats_out = out_dir / f"{p.stem}_stats.json"
                    stats_payload = {
                        'file': str(p.name),
                        'clusters': stats,
                        'config': {
                            'eps': eps,
                            'min_samples': min_samples,
                            'merge_distance': float(cluster_cfg.get('merge_small', {}).get('distance', 0)),
                        },
                    }
                    try:
                        stats_out.write_text(json.dumps(stats_payload, indent=2))
                    except Exception:
                        pass

            # Landmark prediction per tooth instance
            save_dict = dict(
                prob_tooth=prob.astype(np.float32),
                sem2=(prob > float(args.thr)).astype(np.uint8),
                inst=inst_all,
                threshold=args.thr,
            )

            if args.landmarks and lm_backbone is not None:
                from ml.infer.landmark_decode import extract_tooth_clouds, predict_landmarks_batch
                tooth_clouds = extract_tooth_clouds(pts, inst_all)
                if tooth_clouds:
                    lm_results = predict_landmarks_batch(
                        tooth_clouds, lm_backbone, lm_head, device,
                        max_points=args.landmark_max_points,
                    )
                    # Store as arrays: landmark_ids (K,), landmark_coords (K, 6, 3)
                    if lm_results:
                        lm_ids = np.array(sorted(lm_results.keys()), dtype=np.int32)
                        lm_coords = np.stack([lm_results[i] for i in lm_ids], axis=0)
                        save_dict["landmark_ids"] = lm_ids
                        save_dict["landmark_coords"] = lm_coords.astype(np.float32)

            np.savez_compressed(out, **save_dict)
            done += 1
            n_clusters = len(set(int(x) for x in np.unique(inst_all) if x >= 0))
            lm_str = ""
            if args.landmarks and "landmark_ids" in save_dict:
                lm_str = f", landmarks={len(save_dict['landmark_ids'])} teeth"
            print(f'  [OK] {out.name}  (clusters={n_clusters}{lm_str})')

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            errors.append((p.stem, 'CUDA OOM'))
            print(f'  [OOM] {p.stem}: CUDA out of memory, skipped (try lower --offset-max-points)')
            continue
        except Exception as e:
            errors.append((p.stem, str(e)))
            print(f'  [ERROR] {p.stem}: {e}')
            continue

    # --- Summary ---
    elapsed_total = time.time() - t_start
    summary = {
        'total_files': total,
        'skipped_existing': skipped,
        'newly_processed': done,
        'errors': len(errors),
        'elapsed_sec': round(elapsed_total, 1),
        'error_details': errors[:50],
    }
    summary_path = out_dir / '_run_summary.json'
    try:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception:
        pass

    print(f'\n{"="*55}')
    print(f'[DONE] {done} processed, {skipped} skipped, {len(errors)} errors  ({elapsed_total:.0f}s)')
    if errors:
        print(f'[ERRORS] {", ".join(e[0] for e in errors[:10])}{"..." if len(errors) > 10 else ""}')


if __name__ == '__main__':
    main()
