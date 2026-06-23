from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from ml.augment.landmark_transforms import apply_aug
from ml.datasets.landmarks import LandmarkDataset, LandmarkSample
from ml.datasets.landmark_teeth import ToothLandmarkDataset, ToothSample


class PointNetEncoder(nn.Module):
    def __init__(self, input_dim: int = 3, feature_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.mlp(x)
        pooled = torch.max(features, dim=1).values
        return pooled


class LandmarkRegressor(nn.Module):
    def __init__(self, num_landmarks: int, feature_dim: int = 256) -> None:
        super().__init__()
        self.encoder = PointNetEncoder(feature_dim=feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_landmarks * 3),
        )
        self.num_landmarks = num_landmarks

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, N, 3)
        latent = self.encoder(points)
        out = self.head(latent)
        return out.view(points.shape[0], self.num_landmarks, 3)


def collate_fn(batch: Tuple[LandmarkSample | ToothSample, ...]) -> tuple:
    points = torch.stack([sample.points for sample in batch], dim=0)
    targets = torch.stack([sample.landmarks for sample in batch], dim=0)
    if hasattr(batch[0], "mask"):
        aux = torch.stack([sample.mask for sample in batch], dim=0)
    else:
        aux = torch.stack([sample.classes for sample in batch], dim=0)
    meta = [sample.meta for sample in batch]
    return points, targets, aux, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Landmark regression baseline trainer")
    ap.add_argument("--manifest", type=Path, required=True, help="Path to landmark manifest JSON")
    ap.add_argument("--points-upper", type=Path, default=Path("temp_data/infer_upper"))
    ap.add_argument("--points-lower", type=Path, default=Path("temp_data/infer_lower"))
    ap.add_argument("--tooth-manifest", type=Path, help="Per-tooth dataset manifest (enables tooth-level training)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-points", type=int, default=4096)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--augment", action="store_true", help="Enable simple geometric augmentations")
    ap.add_argument(
        "--augment-mode",
        choices=("default", "weak", "strong"),
        default="default",
        help="Augmentation strength preset (requires --augment)",
    )
    ap.add_argument("--loss", choices=("mse", "huber"), default="mse", help="Regression loss function")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.augment and args.augment_mode != "default":
        print("[WARN] --augment-mode ignored because --augment is disabled")

    use_tooth_dataset = bool(args.tooth_manifest)
    if use_tooth_dataset:
        dataset = ToothLandmarkDataset(
            manifest_path=args.tooth_manifest,
            max_points=args.max_points,
        )
    else:
        dataset = LandmarkDataset(
            manifest_path=args.manifest,
            points_upper_root=args.points_upper,
            points_lower_root=args.points_lower,
            max_points=args.max_points,
        )

    val_len = int(len(dataset) * args.val_ratio)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    ) if val_len > 0 else None

    model = LandmarkRegressor(num_landmarks=dataset.landmark_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    def _compute_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if args.loss == "huber":
            per_elem = F.smooth_l1_loss(preds, targets, reduction="none", beta=1.0)
        else:
            per_elem = F.mse_loss(preds, targets, reduction="none")
        if mask is not None:
            weighted = per_elem * mask.unsqueeze(-1)
            denom = torch.clamp(mask.sum() * targets.shape[-1], min=1.0)
            return weighted.sum() / denom
        return per_elem.mean()

    history = []

    def _metrics_mm(
        preds: torch.Tensor,
        targets: torch.Tensor,
        meta_batch: list[dict],
        thresholds: tuple[float, ...],
        mask: torch.Tensor | None,
    ) -> tuple[float, Dict[float, float], int]:
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy() if mask is not None else None
        total_sq = 0.0
        count = 0
        success = {thr: 0.0 for thr in thresholds}
        for i, meta in enumerate(meta_batch):
            scale = np.asarray(meta.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(1, -1)
            diff = (preds_np[i] - targets_np[i]) * scale
            if mask_np is not None:
                active = mask_np[i].reshape(-1, 1)
                diff = diff * active
                valid = active.squeeze(-1) > 0.5
                dist = np.linalg.norm(diff[valid], axis=1)
                total_sq += float(np.sum(dist ** 2))
                count += dist.size
                for thr in thresholds:
                    success[thr] += float(np.count_nonzero(dist <= thr))
            else:
                dist = np.linalg.norm(diff, axis=1)
                total_sq += float(np.sum(dist ** 2))
                count += dist.size
                for thr in thresholds:
                    success[thr] += float(np.count_nonzero(dist <= thr))
        rmse = math.sqrt(total_sq / max(count, 1))
        return rmse, success, count

    thresholds = (2.0, 4.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        rmse_acc = []
        train_success: Dict[float, float] = {thr: 0.0 for thr in thresholds}
        total_landmarks = 0
        for points, targets, aux, meta in train_loader:
            points = points.to(device)
            targets = targets.to(device)
            mask_tensor = None
            if use_tooth_dataset:
                mask_tensor = aux.to(device)
            if args.augment:
                points, targets = apply_aug(points, targets, mode=args.augment_mode)
            optimizer.zero_grad()
            preds = model(points)
            loss = _compute_loss(preds, targets, mask_tensor)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * points.size(0)
            rmse_val, succ_counts, total = _metrics_mm(preds, targets, meta, thresholds, mask_tensor)
            rmse_acc.append(rmse_val)
            for thr, val in succ_counts.items():
                train_success[thr] = train_success.get(thr, 0.0) + val
            total_landmarks += total
        train_loss = running_loss / len(train_set)
        train_rmse = float(sum(rmse_acc) / max(len(rmse_acc), 1))
        train_sr = {thr: (train_success.get(thr, 0.0) / max(total_landmarks, 1)) for thr in train_success}

        val_loss = float("nan")
        val_rmse = float("nan")
        val_sr: Dict[float, float] | None = None
        if val_loader is not None:
            model.eval()
            total = 0.0
            count = 0
            rmse_vals = []
            val_success: Dict[float, float] = {thr: 0.0 for thr in thresholds}
            total_landmarks_val = 0
            with torch.no_grad():
                for points, targets, aux, meta in val_loader:
                    points = points.to(device)
                    targets = targets.to(device)
                    mask_tensor = None
                    if use_tooth_dataset:
                        mask_tensor = aux.to(device)
                    preds = model(points)
                    loss = _compute_loss(preds, targets, mask_tensor)
                    total += loss.item() * points.size(0)
                    count += points.size(0)
                    rmse_val, succ_counts, tot = _metrics_mm(preds, targets, meta, thresholds, mask_tensor)
                    rmse_vals.append(rmse_val)
                    for thr, val in succ_counts.items():
                        val_success[thr] = val_success.get(thr, 0.0) + val
                    total_landmarks_val += tot
            val_loss = total / max(1, count)
            if rmse_vals:
                val_rmse = float(sum(rmse_vals) / len(rmse_vals))
                val_sr = {thr: val_success.get(thr, 0.0) / max(total_landmarks_val, 1) for thr in val_success}

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_rmse_mm": train_rmse,
            "val_loss": val_loss,
            "val_rmse_mm": val_rmse,
            "train_sr_2mm": train_sr.get(2.0),
            "train_sr_4mm": train_sr.get(4.0),
            "val_sr_2mm": (val_sr or {}).get(2.0),
            "val_sr_4mm": (val_sr or {}).get(4.0),
        })
        sr_train_msg = " ".join([f"train_SR@{int(thr)}mm={train_sr.get(thr, float('nan')):.2f}" for thr in sorted(train_sr)])
        sr_val_msg = ""
        if val_sr:
            sr_val_msg = " " + " ".join([f"val_SR@{int(thr)}mm={val_sr.get(thr, float('nan')):.2f}" for thr in sorted(val_sr)])
        print(
            f"[EPOCH {epoch}] train_loss={train_loss:.4f} "
            f"train_rmse={train_rmse:.3f}mm val_loss={val_loss:.4f} val_rmse={val_rmse:.3f}mm"
            f" {sr_train_msg}{sr_val_msg}"
        )

    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "landmark_train_metrics.json"
    out_path.write_text(json.dumps(history, indent=2))
    print(f"[INFO] metrics written to {out_path}")


if __name__ == "__main__":
    main()
