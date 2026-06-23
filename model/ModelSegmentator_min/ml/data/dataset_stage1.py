from __future__ import annotations

import random
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess_features import compute_features, FeatureConfig
from .augment import (random_flip_lr, random_jitter, random_rotate_z,
                      random_scale, random_shift, random_rotate_perturbation,
                      elastic_deformation, random_cutout_sphere)
from ml.utils.exclude import filter_paths

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Stage1DatasetConfig:
    root: Path
    manifest: Path | None = None
    max_points: int = 60000
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    augment: dict = field(default_factory=dict)
    use_cached: bool = True            # if npz has 'feats', use it
    cache_to_npz: bool = False         # if True and feats not present, write *_feats.npz
    limit_train_files: Optional[int] = None
    limit_val_files: Optional[int] = None
    exclude_cases: set[str] = field(default_factory=set)
    subset_list: Path | None = None
    val_subset_list: Path | None = None
    hardcase_roots: list[Path] = field(default_factory=list)
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"


def _scan_npz(root: Path, exclude: set[str] | None = None) -> List[Path]:
    files = []
    for p in root.rglob("*.npz"):
        if p.name.endswith("_feats.npz"):
            continue  # cached feature files do not contain labels
        files.append(p)
    filtered = filter_paths(files, exclude or set())
    return sorted(filtered)


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _resolve_list_item(root: Path, entry: str) -> Optional[Path]:
    if not entry:
        return None
    path = Path(entry)
    if not path.is_absolute():
        path = root / path
    if path.suffix != ".npz":
        resolved = _resolve_case_path(root, entry, {})
        if resolved:
            return resolved
    if path.exists() and path.suffix == ".npz" and not path.name.endswith("_feats.npz"):
        return path.resolve()
    return None


def _paths_from_list(list_path: Path, root: Path, exclude: set[str]) -> List[Path]:
    entries: list[Path] = []
    for raw in list_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        stem = Path(line).stem
        if stem in exclude:
            continue
        path = _resolve_list_item(root, line)
        if path and path.stem not in exclude:
            entries.append(path)
    return _dedupe_paths(entries)


def _resolve_case_path(root: Path, case_id: str, row: dict[str, str]) -> Optional[Path]:
    candidates = [
        root / f"{case_id}_sample-sample.npz",
        root / f"{case_id}.npz",
    ]
    if "_" in case_id:
        suffix = case_id.split("_")[-1]
        candidates.append(root / suffix / f"{case_id}.npz")
        candidates.append(root / suffix / f"{case_id}_sample-sample.npz")
    source_npz = (row.get("source_npz") or "").strip()
    if source_npz:
        src_path = Path(source_npz)
        if not src_path.is_absolute():
            src_path = (_REPO_ROOT / src_path).resolve()
        candidates.append(src_path)
    for cand in candidates:
        if not cand:
            continue
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = Path(cand)
        if resolved.exists() and resolved.suffix == ".npz" and not resolved.name.endswith("_feats.npz"):
            return resolved
    return None


def _paths_from_manifest(cfg: Stage1DatasetConfig, split_label: str) -> List[Path]:
    if not cfg.manifest or not cfg.manifest.exists() or not split_label:
        return []
    paths: list[Path] = []
    seen: set[Path] = set()
    with cfg.manifest.open() as infile:
        reader = csv.DictReader(line for line in infile if not line.lstrip().startswith("#"))
        for row in reader:
            case_id = (row.get("case_id") or row.get("case") or "").strip()
            if not case_id:
                continue
            if case_id in cfg.exclude_cases:
                continue
            row_split = (row.get("split") or row.get("set") or "").strip().lower()
            if row_split != split_label:
                continue
            path = _resolve_case_path(cfg.root, case_id, row)
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
    return sorted(paths)


class Stage1Dataset(Dataset):
    """Binary tooth-vs-gingiva dataset reader.

    Expected NPZ keys per sample:
      - points: (N,3) float32, required
      - normals: (N,3) float32, optional
      - sem2: (N,) int {0,1} (gingiva=0, tooth=1), required
      - boundary: (N,) int {0,1}, optional (defaults to zeros)

    If the sample has more than cfg.max_points, it will be randomly downsampled.
    """

    def __init__(self, cfg: Stage1DatasetConfig, split: str = "train", val_ratio: float = 0.1, seed: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.augment_cfg = cfg.augment or {}
        self.apply_augment = split == "train" and any(
            bool(self.augment_cfg.get(k, False))
            for k in ("rotate", "jitter", "flip", "mirror", "dropout",
                       "scale", "shift", "rotate_perturbation", "elastic", "cutout")
        )
        subset_path = cfg.subset_list if split == "train" else cfg.val_subset_list
        manifest_split = None
        if subset_path:
            self.files = _paths_from_list(subset_path, cfg.root, cfg.exclude_cases)
        elif cfg.manifest and cfg.manifest.exists():
            key = split.lower()
            if key == "train":
                manifest_split = cfg.train_split.lower()
            elif key == "val":
                manifest_split = cfg.val_split.lower()
            elif key == "test":
                manifest_split = cfg.test_split.lower()
            self.files = _paths_from_manifest(cfg, manifest_split or "")
        else:
            all_files = _scan_npz(cfg.root, cfg.exclude_cases)
            if not all_files:
                raise FileNotFoundError(f"No .npz files under {cfg.root}")
            random.Random(seed).shuffle(all_files)
            val_n = max(1, int(len(all_files) * val_ratio))
            if split == "train":
                self.files = all_files[val_n:]
            else:
                self.files = all_files[:val_n]

        if split == "train" and cfg.hardcase_roots:
            extra_files: list[Path] = []
            for root in cfg.hardcase_roots:
                extra_files.extend(_scan_npz(root, cfg.exclude_cases))
            self.files = _dedupe_paths(list(self.files) + extra_files)

        if cfg.manifest and cfg.manifest.exists() and not subset_path:
            if split == "train" and cfg.limit_train_files:
                self.files = self.files[: int(cfg.limit_train_files)]
            if split != "train" and cfg.limit_val_files:
                self.files = self.files[: int(cfg.limit_val_files)]
        elif not subset_path:
            if split == "train" and cfg.limit_train_files:
                self.files = self.files[: int(cfg.limit_train_files)]
            if split != "train" and cfg.limit_val_files:
                self.files = self.files[: int(cfg.limit_val_files)]

        if not self.files:
            raise FileNotFoundError(f"No matching samples for split={split} (manifest={cfg.manifest})")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        data = np.load(path)
        points = data["points"].astype(np.float32)
        normals = data["normals"].astype(np.float32) if "normals" in data else None
        if "sem2" in data:
            sem2 = data["sem2"].astype(np.int64)
        elif "tooth_mask" in data:
            sem2 = data["tooth_mask"].astype(np.int64)
        elif "labels" in data:
            sem2 = (data["labels"].astype(np.int64) > 0).astype(np.int64)
        else:
            raise KeyError(f"{path} missing sem2/tooth_mask")
        boundary = data["boundary"].astype(np.int64) if "boundary" in data else np.zeros((points.shape[0],), dtype=np.int64)
        boundary_w = data["boundary_w"].astype(np.float32) if "boundary_w" in data else boundary.astype(np.float32)
        feats_inline = data["feats"].astype(np.float32) if (self.cfg.use_cached and "feats" in data) else None

        if self.apply_augment:
            if self.augment_cfg.get("rotate", False):
                points, normals = random_rotate_z(points, normals)
            if self.augment_cfg.get("flip", False):
                flip_prob = float(self.augment_cfg.get("flip_prob", 0.5))
                points, normals = random_flip_lr(points, normals, prob=flip_prob)
            if self.augment_cfg.get("mirror", False):
                if np.random.rand() < float(self.augment_cfg.get("mirror_prob", 0.5)):
                    points = points.copy()
                    points[:, 1] *= -1.0
                    if normals is not None:
                        normals = normals.copy()
                        normals[:, 1] *= -1.0
            if self.augment_cfg.get("jitter", False):
                jitter_sigma = float(self.augment_cfg.get("jitter_sigma", 0.005))
                jitter_clip = float(self.augment_cfg.get("jitter_clip", 0.02))
                points = random_jitter(points, sigma=jitter_sigma, clip=jitter_clip)
            if self.augment_cfg.get("scale", False):
                sr = self.augment_cfg.get("scale_range", [0.9, 1.1])
                points = random_scale(points, scale_range=tuple(sr))
            if self.augment_cfg.get("shift", False):
                points = random_shift(points, shift_range=float(self.augment_cfg.get("shift_range", 0.1)))
            if self.augment_cfg.get("rotate_perturbation", False):
                rot_s = float(self.augment_cfg.get("rotate_sigma", 0.03))
                rot_c = float(self.augment_cfg.get("rotate_clip", 0.09))
                points, normals = random_rotate_perturbation(points, normals, angle_sigma=rot_s, angle_clip=rot_c)
            if self.augment_cfg.get("elastic", False):
                el_sigma = float(self.augment_cfg.get("elastic_sigma", 0.05))
                el_grid = int(self.augment_cfg.get("elastic_grid", 4))
                points, normals = elastic_deformation(points, normals, sigma=el_sigma, grid_size=el_grid)
            if self.augment_cfg.get("cutout", False):
                co_n = int(self.augment_cfg.get("cutout_n", 1))
                co_r = self.augment_cfg.get("cutout_radius", [0.05, 0.15])
                points, normals, sem2, boundary, boundary_w = random_cutout_sphere(
                    points, normals, sem2, boundary, boundary_w,
                    n_spheres=co_n, radius_range=tuple(co_r))
            drop_ratio = float(self.augment_cfg.get("dropout", 0.0) or 0.0)
            if drop_ratio > 0.0 and points.shape[0] > 1:
                keep_n = max(32, int(points.shape[0] * (1.0 - drop_ratio)))
                keep_n = min(points.shape[0], keep_n)
                sel = np.random.choice(points.shape[0], size=keep_n, replace=False)
                points = points[sel]
                if normals is not None:
                    normals = normals[sel]
                sem2 = sem2[sel]
                boundary = boundary[sel]
                boundary_w = boundary_w[sel]
            feats_inline = None
        # downsample (with optional boundary-aware oversampling)
        N = points.shape[0]
        idxs = None
        if self.cfg.max_points and N > self.cfg.max_points:
            rng = np.random.default_rng(idx)
            boundary_oversample = float(self.augment_cfg.get('boundary_oversample', 1.0))
            if boundary_oversample > 1.0 and boundary.sum() > 0:
                weights = np.ones(N, dtype=np.float64)
                weights[boundary > 0] = boundary_oversample
                weights /= weights.sum()
                idxs = rng.choice(N, size=self.cfg.max_points, replace=False, p=weights)
            else:
                idxs = rng.choice(N, size=self.cfg.max_points, replace=False)

        def _sel(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if arr is None:
                return None
            return arr[idxs] if idxs is not None else arr

        points = _sel(points)
        normals = _sel(normals) if normals is not None else None
        sem2 = _sel(sem2)
        boundary = _sel(boundary)
        boundary_w = _sel(boundary_w)

        if feats_inline is not None:
            feats = _sel(feats_inline)
        else:
            feats = compute_features(points, normals, self.cfg.feature)
            if self.cfg.cache_to_npz:
                out = path.with_name(path.stem + "_feats.npz")
                try:
                    np.savez_compressed(out,
                        points=points,
                        normals=(normals if normals is not None else np.zeros_like(points, dtype=np.float32)),
                        sem2=sem2,
                        boundary=boundary,
                        boundary_w=boundary_w,
                        feats=feats,
                    )
                except Exception:
                    pass
        return {
            "feats": torch.from_numpy(feats),         # (N,F)
            "sem2": torch.from_numpy(sem2),           # (N,)
            "boundary": torch.from_numpy(boundary),   # (N,)
            "boundary_w": torch.from_numpy(boundary_w.astype(np.float32)),
        }
