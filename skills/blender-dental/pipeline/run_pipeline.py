"""中空オープン模型 一括パイプライン（STL投入 → 中空オープンSTL出力）。

工程: スキャン読込 → 歯/歯肉セグメンテーション(ML) → 歯＋歯肉バンド抽出 →
      口蓋カットを放物線フィットで平滑化 → 簡略化 → 咬合平面に垂直な土台リム延長
      → ブーリアン中空化（外形保持・自己交差なし・watertight）→ 底開放 → 書出し。

確定パラメータ（2026-06-22 先生承認）:
  band_mm=5 / keep_pct=40(口蓋カラー幅) / degree=4(放物線) / tooth_safety=0.8(歯保護)
  rim_mm=3 / wall_mm=2 / pitch=0.2 / open_bottom=True

CLI:
  python3 run_pipeline.py --in SCAN.stl --out model.stl --arch upper
  オプション: --closed（底を閉じた中空に）, --solid（中空にせず中実土台）
依存: torch(cpu可), trimesh, scipy, scikit-learn, fast_simplification,
      pymeshfix, manifold3d, mapbox_earcut, および Model Segmentator リポジトリ
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import fast_simplification as fs
import numpy as np
import trimesh
from scipy.spatial import cKDTree

import geometry_ops as g
import ml_segment as ms

REPO = Path("/Users/ema/Desktop/VScode/Model Segmentator")


def run(inp: str, out: str, arch: str = "upper", *, band_mm: float = 5.0,
        keep_pct: float = 40.0, degree: int = 4, tooth_safety_mm: float = 0.8,
        target_faces: int = 60000, rim_mm: float = 3.0, wall_mm: float = 2.0,
        pitch: float = 0.2, mode: str = "open", repo: Path = REPO) -> dict:
    """フルパイプラインを実行し、最終STLを書き出して情報dictを返す。"""
    t0 = time.time()
    mesh = trimesh.load(inp, process=True)
    mesh.merge_vertices()

    # 1) 歯/歯肉セグメンテーション → 2) 歯＋歯肉バンド抽出
    prob = ms.segment(mesh, arch=arch, repo=repo, verbose=False)
    band, _ = ms.extract_with_gum_band(mesh, prob, thr=0.5, band_mm=band_mm)

    # 3) 口蓋カットを放物線フィットで平滑化（probをバンド頂点へ最近傍移植）
    _, idx = cKDTree(mesh.vertices).query(band.vertices)
    band2, cut_info = ms.smooth_palatal_cut(
        band, prob[idx], thr=0.5, margin_mm=0.5, degree=degree,
        keep_pct=keep_pct, tooth_safety_mm=tooth_safety_mm)

    # 4) 簡略化（ボクセル化を高速化するため）
    vc, fc = fs.simplify(
        np.asarray(band2.vertices, np.float32),
        np.asarray(band2.faces, np.int32), target_count=target_faces)
    surf = trimesh.Trimesh(vc, fc, process=True)

    # 5) 土台リム延長 →（中空化／中実）→ 書出し
    if mode == "solid":
        result, info = g.extrude_base(surf, depth=rim_mm)
    else:
        result, info = g.make_hollow_open_model(
            surf, rim_mm=rim_mm, wall_mm=wall_mm, pitch=pitch,
            open_bottom=(mode == "open"))
    result.export(out)

    info.update(arch=arch, mode=mode, cut=cut_info,
                seconds=round(time.time() - t0, 1), out=out)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="中空オープン模型 一括パイプライン")
    ap.add_argument("--in", dest="inp", required=True, help="入力スキャンSTL")
    ap.add_argument("--out", dest="out", required=True, help="出力STL")
    ap.add_argument("--arch", default="upper", choices=["upper", "lower"])
    ap.add_argument("--rim", type=float, default=3.0, help="土台リム高さmm")
    ap.add_argument("--wall", type=float, default=2.0, help="肉厚mm")
    ap.add_argument("--pitch", type=float, default=0.2, help="内部キャビティ解像度mm")
    ap.add_argument("--keep-pct", type=float, default=40.0, help="口蓋カラー幅")
    ap.add_argument("--repo", default=str(REPO), help="Model Segmentatorの場所")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--closed", action="store_true", help="底を閉じた中空にする")
    grp.add_argument("--solid", action="store_true", help="中空にせず中実土台にする")
    a = ap.parse_args()

    mode = "solid" if a.solid else "closed" if a.closed else "open"
    info = run(a.inp, a.out, a.arch, keep_pct=a.keep_pct, rim_mm=a.rim,
               wall_mm=a.wall, pitch=a.pitch, mode=mode, repo=Path(a.repo))
    print("DONE:", info)


if __name__ == "__main__":
    main()
