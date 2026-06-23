from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import numpy as np
import torch
import yaml

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from ml.models.backbone_pointnet2 import DummyBackbone
from ml.models.pointnet2_backbone import PointNet2BackboneLite
from ml.models.heads import HeadSem2
from ml.infer.smooth import knn_laplacian_smooth


def load_ckpt(ckpt_path: Path, in_channels: int, device, backbone_type: str = "pointnet2_lite") -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load checkpoint with specified backbone type."""
    if backbone_type == "pointnet2_lite":
        bb = PointNet2BackboneLite(in_channels=in_channels, out_dim=128).to(device)
    else:
        bb = DummyBackbone(in_channels=in_channels, out_dim=128).to(device)
    hd = HeadSem2(in_dim=128).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    bb.load_state_dict(ck.get("backbone", {}), strict=False)
    hd.load_state_dict(ck.get("head_sem2", {}), strict=False)
    bb.eval(); hd.eval()
    return bb, hd


def rotate_z(xyz: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate points around Z-axis."""
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return xyz @ rot.T


def predict_with_tta(
    feats: np.ndarray,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    device: torch.device,
    tta_angles: List[float] = None
) -> np.ndarray:
    """
    Predict with Test Time Augmentation (TTA).

    Args:
        feats: (N, C) input features, first 3 are xyz, 3:6 are normals
        backbone: backbone model
        head: head model
        device: torch device
        tta_angles: list of rotation angles in degrees. None for no TTA.

    Returns:
        probs: (N,) tooth probability for each point
    """
    if tta_angles is None or len(tta_angles) <= 1:
        # No TTA
        x = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            f = backbone(x)
            logits = head(f)
            probs = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()
        return probs

    # TTA with multiple rotations
    xyz = feats[:, :3]
    normals = feats[:, 3:6] if feats.shape[1] >= 6 else None
    other = feats[:, 6:] if feats.shape[1] > 6 else None

    probs_sum = np.zeros(len(feats), dtype=np.float32)

    with torch.no_grad():
        for angle in tta_angles:
            # Rotate xyz and normals
            xyz_rot = rotate_z(xyz, angle)
            if normals is not None:
                normals_rot = rotate_z(normals, angle)
                if other is not None:
                    feats_rot = np.concatenate([xyz_rot, normals_rot, other], axis=1)
                else:
                    feats_rot = np.concatenate([xyz_rot, normals_rot], axis=1)
            else:
                feats_rot = xyz_rot if other is None else np.concatenate([xyz_rot, other], axis=1)

            x = torch.from_numpy(feats_rot.astype(np.float32)).unsqueeze(0).to(device)
            f = backbone(x)
            logits = head(f)
            probs_sum += torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()

    return probs_sum / len(tta_angles)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(_repo / "configs/stage1.yaml"))
    ap.add_argument("--input", type=str, default=str(_repo / "data/sample_npz/upper"))
    ap.add_argument("--ckpt", type=str, default=str(_repo / "ckpts/stage1_last.pth"))
    ap.add_argument("--out", type=str, default=str(_repo / "pred/stage1"))
    ap.add_argument("--backbone", type=str, default="pointnet2_lite",
                    choices=["pointnet2_lite", "dummy"], help="Backbone type")
    ap.add_argument("--tta", type=int, default=8,
                    help="Number of TTA rotations (0=disabled, 4 or 8 recommended)")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="Classification threshold (default: 0.90 optimized)")
    ap.add_argument("--smooth", action="store_true", default=True,
                    help="Apply KNN Laplacian smoothing")
    ap.add_argument("--no-smooth", dest="smooth", action="store_false")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit number of files to process (0=all)")
    ap.add_argument("--refine", action="store_true",
                    help="Apply bilateral boundary refinement")
    ap.add_argument("--refine-k", type=int, default=16,
                    help="KNN neighbors for boundary refinement")
    ap.add_argument("--refine-iters", type=int, default=3,
                    help="Smoothing iterations for boundary refinement")
    ap.add_argument("--refine-alpha", type=float, default=0.3,
                    help="Smoothing strength for boundary refinement")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    in_dir = Path(args.input)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt)

    files = sorted(in_dir.glob("*.npz"))
    if not files:
        print(f"[WARN] no npz under {in_dir}")
        return

    if args.limit > 0:
        files = files[:args.limit]

    # Infer in_channels from first file
    sample = np.load(files[0])
    feats0 = sample.get("feats")
    if feats0 is None:
        from ml.data.preprocess_features import compute_features
        feats0 = compute_features(sample["points"].astype(np.float32), sample.get("normals"))
    in_channels = feats0.shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    bb, hd = load_ckpt(ckpt, in_channels, device, backbone_type=args.backbone)

    # Setup TTA angles
    if args.tta <= 0:
        tta_angles = None
        print(f"[INFO] TTA disabled")
    elif args.tta == 4:
        tta_angles = [0, 90, 180, 270]
        print(f"[INFO] TTA enabled: 4 rotations")
    elif args.tta == 8:
        tta_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        print(f"[INFO] TTA enabled: 8 rotations")
    else:
        tta_angles = [360.0 * i / args.tta for i in range(args.tta)]
        print(f"[INFO] TTA enabled: {args.tta} rotations")

    print(f"[INFO] Backbone: {args.backbone}")
    print(f"[INFO] Threshold: {args.threshold}")
    print(f"[INFO] Smoothing: {args.smooth}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Processing {len(files)} files...")

    for i, p in enumerate(files):
        data = np.load(p)
        feats = data.get("feats")
        if feats is None:
            from ml.data.preprocess_features import compute_features
            feats = compute_features(data["points"].astype(np.float32), data.get("normals"))

        # Predict with TTA
        prob = predict_with_tta(feats, bb, hd, device, tta_angles)

        # Optional bilateral boundary refinement
        if args.refine:
            from ml.infer.boundary_refine import bilateral_boundary_smooth
            pts_r = data["points"].astype(np.float32)
            nrm_r = data.get("normals")
            if nrm_r is not None:
                nrm_r = nrm_r.astype(np.float32)
            prob = bilateral_boundary_smooth(prob, pts_r, nrm_r,
                                             k=args.refine_k, iters=args.refine_iters,
                                             alpha=args.refine_alpha)

        # Optional smoothing
        if args.smooth:
            pts = data["points"].astype(np.float32)
            prob_s = knn_laplacian_smooth(prob, pts, k=16, iters=5, lam=0.5)
        else:
            prob_s = prob

        # Apply optimized threshold
        pred = (prob_s >= args.threshold).astype(np.int32)

        out = out_dir / (p.stem + "_pred.npz")
        np.savez_compressed(out, pred=pred, prob=prob, prob_s=prob_s, threshold=args.threshold)

        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"[{i+1}/{len(files)}] {out.name}")

    print(f"[DONE] Output saved to {out_dir}")


if __name__ == "__main__":
    main()
