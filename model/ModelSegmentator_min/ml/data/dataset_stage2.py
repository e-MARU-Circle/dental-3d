from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess_features import compute_features, FeatureConfig
from ml.utils.exclude import filter_paths

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Stage2DatasetConfig:
    root: Path
    manifest: Path | None = None
    max_points: int = 60000
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    use_cached: bool = True
    cache_to_npz: bool = False
    limit_train_files: int | None = None
    limit_val_files: int | None = None
    exclude_cases: set[str] = field(default_factory=set)
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"


def _scan_npz(root: Path, exclude: set[str] | None = None) -> List[Path]:
    # Exclude cached feature files to avoid double-counting samples
    files = [p for p in root.rglob('*.npz') if p.is_file() and not p.name.endswith('_feats.npz')]
    return sorted(filter_paths(files, exclude or set()))


def _resolve_case_path(root: Path, case_id: str, row: dict[str, str]) -> Path | None:
    candidates = [
        root / f"{case_id}.npz",
        root / f"{case_id}_sample-sample.npz",
    ]
    if "_" in case_id:
        suffix = case_id.split('_')[-1]
        candidates.append(root / suffix / f"{case_id}.npz")
        candidates.append(root / suffix / f"{case_id}_sample-sample.npz")
    source_npz = (row.get('source_npz') or '').strip()
    if source_npz:
        p = Path(source_npz)
        if not p.is_absolute():
            p = (_REPO_ROOT / p).resolve()
        candidates.append(p)
    for cand in candidates:
        if not cand:
            continue
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = Path(cand)
        if resolved.exists() and resolved.suffix == '.npz' and not resolved.name.endswith('_feats.npz'):
            return resolved
    return None


def _paths_from_manifest(cfg: Stage2DatasetConfig, split_label: str) -> List[Path]:
    if not cfg.manifest or not cfg.manifest.exists() or not split_label:
        return []
    paths: list[Path] = []
    seen: set[Path] = set()
    with cfg.manifest.open() as infile:
        reader = csv.DictReader(line for line in infile if not line.lstrip().startswith('#'))
        for row in reader:
            case_id = (row.get('case_id') or row.get('case') or '').strip()
            if not case_id or case_id in cfg.exclude_cases:
                continue
            row_split = (row.get('split') or row.get('set') or '').strip().lower()
            if row_split != split_label:
                continue
            path = _resolve_case_path(cfg.root, case_id, row)
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
    return sorted(paths)


class Stage2Dataset(Dataset):
    """Embedding学習用データセット。

    期待NPZキー:
      points(N,3), normals(N,3), tooth_mask(N,), inst(N,) ほか（feats optional）
    歯点のみを使用して特徴/ラベルを返す。
    """

    def __init__(self, cfg: Stage2DatasetConfig, split: str = 'train', val_ratio: float = 0.1, seed: int = 0) -> None:
        self.cfg = cfg
        manifest_split = None
        if cfg.manifest and cfg.manifest.exists():
            key = split.lower()
            if key == 'train':
                manifest_split = cfg.train_split.lower()
            elif key == 'val':
                manifest_split = cfg.val_split.lower()
            elif key == 'test':
                manifest_split = cfg.test_split.lower()
            self.files = _paths_from_manifest(cfg, manifest_split or '')
        else:
            files = _scan_npz(cfg.root, cfg.exclude_cases)
            if not files:
                raise FileNotFoundError(f"No npz under {cfg.root}")
            import random
            random.Random(seed).shuffle(files)
            val_n = max(1, int(len(files) * val_ratio))
            if split == 'train':
                self.files = files[val_n:]
                if cfg.limit_train_files:
                    self.files = self.files[: int(cfg.limit_train_files)]
            else:
                self.files = files[:val_n]
                if cfg.limit_val_files:
                    self.files = self.files[: int(cfg.limit_val_files)]

        if cfg.manifest and cfg.manifest.exists():
            if split == 'train' and cfg.limit_train_files:
                self.files = self.files[: int(cfg.limit_train_files)]
            if split != 'train' and cfg.limit_val_files:
                self.files = self.files[: int(cfg.limit_val_files)]

        if not self.files:
            raise FileNotFoundError(f"No matching samples for split={split} (manifest={cfg.manifest})")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        d = np.load(p)
        pts = d['points'].astype(np.float32)
        nrm = d['normals'].astype(np.float32) if 'normals' in d else None
        tooth = d['tooth_mask'].astype(np.int64) if 'tooth_mask' in d else d['sem2'].astype(np.int64)
        inst = d['inst'].astype(np.int64) if 'inst' in d else np.zeros((pts.shape[0],), dtype=np.int64)
        # select tooth points only (default)
        sel = (tooth > 0)
        if not np.any(sel):
            sel = np.ones((pts.shape[0],), dtype=bool)

        cache_path = p.with_name(p.stem + '_feats.npz')
        use_cached_file = bool(self.cfg.use_cached and (('feats' not in d) and cache_path.exists()))

        if use_cached_file:
            dc = np.load(cache_path)
            pts = dc['points'].astype(np.float32)
            nrm = dc['normals'].astype(np.float32) if 'normals' in dc else None
            inst = dc['inst'].astype(np.int64)
        else:
            # apply selection to original
            pts = pts[sel]
            nrm = (nrm[sel] if nrm is not None else None)
            inst = inst[sel]

        # downsample first
        N = pts.shape[0]
        idxs = None
        if self.cfg.max_points and N > self.cfg.max_points:
            idxs = np.random.default_rng(0).choice(N, size=self.cfg.max_points, replace=False)
            pts = pts[idxs]
            nrm = (nrm[idxs] if nrm is not None else None)
            inst = inst[idxs]

        # build feats after downsample
        if use_cached_file:
            feats_full = dc['feats'].astype(np.float32)
            if idxs is not None:
                feats_full = feats_full[idxs]
        elif (self.cfg.use_cached and 'feats' in d):
            feats_full = d['feats'].astype(np.float32)[sel]
            if idxs is not None:
                feats_full = feats_full[idxs]
        else:
            feats_full = compute_features(pts, nrm, self.cfg.feature)
            # optional on-disk caching for subsequent runs (tooth-only, post-downsample)
            if self.cfg.cache_to_npz:
                out = p.with_name(p.stem + '_feats.npz')
                try:
                    np.savez_compressed(out,
                        points=pts,
                        normals=(nrm if nrm is not None else np.zeros_like(pts, dtype=np.float32)),
                        tooth_mask=np.ones((pts.shape[0],), dtype=np.int64),
                        inst=inst,
                        feats=feats_full)
                except Exception:
                    pass
        feats = feats_full.astype(np.float32)
        return {
            'feats': torch.from_numpy(feats),   # (M,F)
            'inst': torch.from_numpy(inst),     # (M,)
            'points': torch.from_numpy(pts.astype(np.float32)),  # (M,3)
        }
