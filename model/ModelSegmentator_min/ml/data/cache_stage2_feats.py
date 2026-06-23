from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from ml.data.dataset_stage2 import Stage2DatasetConfig, Stage2Dataset
from ml.data.preprocess_features import FeatureConfig
from ml.utils.exclude import read_exclude_list


def main() -> None:
    ap = argparse.ArgumentParser(description='Precompute and cache Stage-2 tooth-only features as *_feats.npz')
    ap.add_argument('--root', type=str, required=True, help='directory containing *.npz samples')
    ap.add_argument('--max-points', type=int, default=30000)
    ap.add_argument('--knn', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0, help='limit number of files (0=all)')
    ap.add_argument('--exclude-list', type=str, default=str(_repo / 'configs/exclude_cases.txt'))
    args = ap.parse_args()

    exclude_path = Path(args.exclude_list)
    if not exclude_path.is_absolute():
        exclude_path = (_repo / exclude_path).resolve()
    exclude_cases = read_exclude_list(exclude_path)

    cfg = Stage2DatasetConfig(
        root=Path(args.root),
        max_points=int(args.max_points),
        feature=FeatureConfig(knn=int(args.knn)),
        use_cached=False,
        cache_to_npz=True,
        limit_train_files=int(args.limit) or None,
        exclude_cases=exclude_cases,
    )
    ds = Stage2Dataset(cfg, split='train', val_ratio=0.0)
    print(f"[INFO] Caching features for {len(ds)} samples under {args.root} ...")
    for i in range(len(ds)):
        _ = ds[i]  # triggers compute + cache if missing
        if (i+1) % 20 == 0:
            print(f"[OK] {i+1}/{len(ds)}")
    print("[DONE] caching complete.")


if __name__ == '__main__':
    main()
