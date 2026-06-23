"""工程2: 歯/歯肉セグメンテーション（研究モデル Model Segmentator / stage1 を利用）。
口蓋・歯肉を除去し、馬蹄形の歯列サーフェスを得る。元メッシュの面・座標は保持。

重要: stage1は生xyzを特徴に使うため、入力を学習フレームへ正準化する必要がある。
  正準化 = 180°(X軸)補正 → XY中心化 → 上端ZをZTOP(-78)へ。
  これを外すと歯認識率がほぼ0%になる（iTero/OrthoCAD書き出しは別座標系のため）。

CLI:
  python3 ml_segment.py --in SCAN.stl --out tooth_only.stl --arch upper --thr 0.5
依存: torch(cpu可), trimesh, scipy, scikit-learn, および Model Segmentator リポジトリ
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import trimesh

REPO = Path("/Users/ema/Desktop/VScode/Model Segmentator")  # 研究モデルの所在

# 向き候補（剛体回転）。上顎/下顎・メーカー差で咬合軸の向き・符号が異なるため、
# どの軸が咬合軸でも拾えるよう離散回転を網羅。Z軸まわりは学習時rotate_z増強で
# 不変なので省略（=Z180も不要）。最も歯と認識される向きを自動採用する。
def _rot(axis, deg):
    t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t)
    if axis == 'x': R = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == 'y': R = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else: R = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return np.array(R, np.float32)

CANDIDATE_ROTS = [
    ("I", np.eye(3, dtype=np.float32)),
    ("X180", _rot('x', 180)), ("Y180", _rot('y', 180)),
    ("X90", _rot('x', 90)), ("X-90", _rot('x', -90)),
    ("Y90", _rot('y', 90)), ("Y-90", _rot('y', -90)),
]


def apply_frame(P, N, R, ztop=-78.0):
    """回転R適用→XY中心化→上端Zをztopへ。剛体変換ゆえ距離保存（原座標kNN伝播可）。"""
    P = (P @ R.T).copy(); N = N @ R.T
    P[:, 0] -= P[:, 0].mean(); P[:, 1] -= P[:, 1].mean()
    P[:, 2] += (ztop - P[:, 2].max())
    return P, N


def segment(mesh: trimesh.Trimesh, arch="upper", thr=0.5, n_inf=40000, repo: Path = REPO,
            verbose=True):
    """auto-orient: 候補向きを総当たりし、最も歯と認識される向きを採用して歯確率を返す。"""
    sys.path.insert(0, str(repo))
    import torch
    from scipy.spatial import cKDTree
    from ml.data.preprocess_features import compute_features, FeatureConfig
    from ml.infer.segment_and_instance import _load_stage1, _infer_sem2

    V = np.asarray(mesh.vertices, np.float32)
    Nrm = np.asarray(mesh.vertex_normals, np.float32)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(V), n_inf, replace=False) if len(V) > n_inf else np.arange(len(V))
    Vs, Ns = V[idx], Nrm[idx]

    ck = repo / "ckpts/stage1_last.pth"
    state = torch.load(ck, map_location="cpu")
    fr = (state.get("config", {}) or {}).get("model", {}).get("feat", {}) or {}
    fc = FeatureConfig(**{k: fr[k] for k in fr if k in FeatureConfig.__annotations__})
    dev = torch.device("cpu")
    bb1 = head1 = None
    best = None  # (score, name, prob_sub)
    for name, R in CANDIDATE_ROTS:
        Pc, Nc = apply_frame(Vs, Ns, R)
        feats = compute_features(Pc, Nc, fc)
        if bb1 is None:
            bb1, head1 = _load_stage1(int(feats.shape[-1]), ck, dev, state=state,
                                      backbone_type="pointnet2_lite")
        prob, _ = _infer_sem2(feats, bb1, head1, dev, thr=thr, tta_angles=None)
        prob = np.asarray(prob)
        score = float((prob >= 0.9).mean())   # 高確信で歯と判定される割合
        if verbose:
            print(f"  [orient {name}] confident-tooth%={100*score:.1f}")
        if best is None or score > best[0]:
            best = (score, name, prob)
    if verbose:
        print(f"[auto-orient] selected={best[1]}  score={100*best[0]:.1f}%")
    prob_sub = best[2]
    _, nn = cKDTree(Vs).query(V, k=1)
    return prob_sub[nn]   # 全頂点の歯確率


def extract_with_gum_band(mesh, prob, thr=0.5, band_mm=5.0, min_keep=2):
    """歯＋『歯頸境界から歯肉側band_mm以内』の歯肉を残す（口蓋など遠位歯肉は除去）。
    band_mmはユークリッド近距離（5mm程度なら曲面でも測地距離に近い）。"""
    from scipy.spatial import cKDTree
    F = np.asarray(mesh.faces); V = np.asarray(mesh.vertices)
    tooth_v = prob >= thr
    tri = tooth_v[F]
    mixed = tri.any(axis=1) & ~tri.all(axis=1)        # 歯/歯肉が混在する面＝歯頸境界
    bnd_v = np.unique(F[mixed].reshape(-1)) if mixed.any() else np.array([], int)
    if len(bnd_v):
        d, _ = cKDTree(V[bnd_v]).query(V)
    else:
        d = np.full(len(V), 1e9)
    keep = tooth_v | (~tooth_v & (d <= band_mm))
    face_keep = keep[F].sum(axis=1) >= min_keep
    return mesh.submesh([np.where(face_keep)[0]], append=True), keep


def smooth_palatal_cut(mesh, prob, thr: float = 0.5,
                       margin_mm: float = 0.5, degree: int = 4,
                       keep_pct: float = 40.0, tooth_safety_mm: float = 0.8):
    """口蓋側のカットラインを、現状の口蓋縁点群に直接フィットした放物線状の
    滑らかな曲線で切り直す。頬・唇側は保持。臼歯部の弓の広がりにも追従する。

    手順: 咬合平面でPCA整列(x=L-R, y=A-P) → 歯頂点で弓向きを判定 → 現状の口蓋
    境界縁点に y=f(x) の多項式(degree)を最小二乗フィット(=波打ちを平滑化) →
    その曲線より口蓋奥(+margin_mm)の面を除去。新しい縁=滑らかな曲線になる。
    """
    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.faces)
    tooth_v = prob >= thr
    # 咬合平面法線(PCA最小軸)→面内をPCA主軸で L-R(x)/A-P(y) に整列
    w, vec = np.linalg.eigh(np.cov(V.T))
    bd = vec[:, 0]
    Pl = V - V.mean(axis=0)
    pln = Pl - np.outer(Pl @ bd, bd)
    w2, v2 = np.linalg.eigh(np.cov(pln.T))
    ax = v2[:, np.argsort(w2)[::-1]]
    x = pln @ ax[:, 0]
    y = pln @ ax[:, 1]

    def _rms(a, b, m):
        return np.sqrt(np.mean((b[m] - np.polyval(np.polyfit(a[m], b[m], 2), a[m])) ** 2))

    if _rms(y, x, tooth_v) < _rms(x, y, tooth_v):    # 弓が x=f(y) に合うなら軸入替
        x, y = y, x
    # 口蓋(凹)側の符号：歯ミッドラインより口蓋側がプラスになるよう
    cy2 = np.polyfit(x[tooth_v], y[tooth_v], 2)
    resid = y - np.polyval(cy2, x)
    s = float(np.sign(np.mean(resid)))
    # 現状の口蓋境界縁点を抽出（境界 ∩ 歯より口蓋側）
    edges = mesh.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    bnd_v = np.unique(unique[counts == 1])
    pal_resid = s * resid
    tooth_med = np.percentile(pal_resid[tooth_v], 90)
    pal_edge = bnd_v[pal_resid[bnd_v] > tooth_med]   # 口蓋側の縁頂点
    # 縁点に放物線状の曲線 y=f(x) をフィット（形状）→ 縁の内側(低percentile)へ寄せ、
    # その滑らかな閾値曲線より口蓋奥を除去（=新しい縁が滑らかな曲線になる）
    cfit = np.polyfit(x[pal_edge], y[pal_edge], degree)
    gap = s * (y[pal_edge] - np.polyval(cfit, x[pal_edge]))
    offset = float(np.percentile(gap, keep_pct))     # 縁の内側へオフセット
    y_thresh = np.polyval(cfit, x) + s * offset
    keep_v = s * (y - y_thresh) <= float(margin_mm)  # 閾値曲線より頬側を保持
    # --- 歯の安全策（二重化）---
    # (1) 歯から tooth_safety_mm 以内の頂点を保護バンドとして無条件保持
    from scipy.spatial import cKDTree
    if tooth_v.any() and tooth_safety_mm > 0:
        dist, _ = cKDTree(V[tooth_v]).query(V)
        protect = tooth_v | (dist <= float(tooth_safety_mm))
    else:
        protect = tooth_v.copy()
    keep_v |= protect
    # (2) 歯(保護)頂点を1つでも含む面は無条件で残す → 歯のメッシュは一切切らない
    face_touch_tooth = protect[F].any(axis=1)
    face_keep = (keep_v[F].sum(axis=1) >= 2) | face_touch_tooth
    out = mesh.submesh([np.where(face_keep)[0]], append=True)
    return out, dict(degree=degree, keep_pct=keep_pct, offset=round(offset, 2),
                     tooth_safety_mm=tooth_safety_mm, n_edge=int(len(pal_edge)),
                     poly=[round(float(c), 5) for c in cfit],
                     kept_faces=int(face_keep.sum()), total_faces=int(len(F)))


def split_tooth(mesh, prob, thr=0.5, min_tooth_verts=2):
    F = np.asarray(mesh.faces)
    tooth_v = prob >= thr
    face_tooth = tooth_v[F].sum(axis=1) >= min_tooth_verts
    tooth = mesh.submesh([np.where(face_tooth)[0]], append=True)
    return tooth, face_tooth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--arch", default="upper", choices=["upper", "lower"])
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--repo", default=str(REPO))
    a = ap.parse_args()
    m = trimesh.load(a.inp, process=True); m.merge_vertices()
    prob = segment(m, a.arch, a.thr, repo=Path(a.repo))
    tooth, ft = split_tooth(m, prob, a.thr)
    tooth.export(a.out)
    print(f"tooth%={100*(prob>=a.thr).mean():.1f}  tooth_faces={ft.sum()}/{len(m.faces)}  -> {a.out}")


if __name__ == "__main__":
    main()
