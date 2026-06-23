from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from ml.models.backbone_pointnet2 import DummyBackbone
from ml.models.heads import HeadEmb
from ml.data.preprocess_features import compute_features


def try_dbscan(x: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    try:
        from sklearn.cluster import DBSCAN  # type: ignore
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
        return lab.astype(np.int32)
    except Exception:
        # simple radius graph BFS as fallback
        from scipy.spatial import cKDTree as KDTree  # type: ignore
        tree = KDTree(x)
        N = x.shape[0]
        labels = -np.ones((N,), dtype=np.int32)
        cid = 0
        for i in range(N):
            if labels[i] >= 0:
                continue
            idx = tree.query_ball_point(x[i], r=eps)
            if len(idx) < min_samples:
                labels[i] = -1
                continue
            # BFS
            stack = list(idx)
            labels[stack[0]] = cid
            while stack:
                u = stack.pop()
                if labels[u] < 0:
                    labels[u] = cid
                for v in tree.query_ball_point(x[u], r=eps):
                    if labels[v] < 0:
                        labels[v] = cid
                        stack.append(v)
            cid += 1
        return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default=str(_repo / 'configs/stage2.yaml'))
    ap.add_argument('--input', type=str, default=str(_repo / 'data/sample_npz/upper'))
    ap.add_argument('--ckpt', type=str, default=str(_repo / 'ckpts/stage2_last.pth'))
    ap.add_argument('--out', type=str, default=str(_repo / 'pred/stage2'))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    in_dir = Path(args.input)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob('*.npz'))
    if not files:
        print(f"[WARN] no npz under {in_dir}")
        return
    # infer in_channels
    sample = np.load(files[0])
    feats0 = sample.get('feats')
    if feats0 is None:
        feats0 = compute_features(sample['points'].astype(np.float32), sample.get('normals'))
    in_channels = feats0.shape[-1]

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'))
    bb = DummyBackbone(in_channels=in_channels, out_dim=128).to(device)
    head = HeadEmb(in_dim=128, out_dim=int(cfg['model'].get('emb_dim', 128))).to(device)
    state = torch.load(Path(args.ckpt), map_location=device)
    bb.load_state_dict(state.get('backbone', {}))
    head.load_state_dict(state.get('head_emb', {}))
    bb.eval(); head.eval()

    eps = float(cfg['cluster']['eps']); min_samples = int(cfg['cluster']['min_samples'])
    for p in files[:16]:
        d = np.load(p)
        pts = d['points'].astype(np.float32)
        nrm = d.get('normals')
        tooth = d.get('tooth_mask', d['sem2']).astype(np.int64)
        sel = (tooth > 0)
        if not np.any(sel):
            sel = np.ones((pts.shape[0],), dtype=bool)
        feats = d.get('feats')
        if feats is None:
            feats = compute_features(pts, nrm)
        x = torch.from_numpy(feats[sel].astype(np.float32)).to(device)
        with torch.no_grad():
            f = bb(x)
            emb = head(f).cpu().numpy()
        # standardize
        mu = emb.mean(axis=0, keepdims=True)
        std = emb.std(axis=0, keepdims=True) + 1e-6
        z = (emb - mu) / std
        lab = try_dbscan(z, eps=eps, min_samples=min_samples)
        inst_all = -np.ones((pts.shape[0],), dtype=np.int32)
        inst_all[sel] = lab
        out = out_dir / (p.stem + '_inst.npz')
        np.savez_compressed(out, inst=inst_all)
        print(f"[OK] {out}")


if __name__ == '__main__':
    main()

