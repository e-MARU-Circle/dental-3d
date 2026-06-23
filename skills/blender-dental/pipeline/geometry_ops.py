"""
中空模型パイプライン 幾何処理（ML抜き / M1）
オープンシェルSTL → 穴補修 → 押し出し(厚み付け) → 中空化

方式: ボクセルベースのシェル化。
  1. voxelize().fill() でソリッド化（=オープンシェルの穴/縁を補修しつつ中身を充填）
  2. binary_erosion で内殻を作り、solid - inner = 一定肉厚の中空シェル
  3. marching cubes でメッシュ化、(任意)Taubin平滑
境界ループが多数あるサーフェスでも側壁生成不要で頑健。模型ベース用途向け。

CLI:
  python3 geometry_ops.py --in IN.stl --out OUT.stl --pitch 0.35 --wall 2.0 --smooth 8
依存: trimesh, scipy, scikit-image, numpy
"""
from __future__ import annotations
import argparse
from collections import Counter
import numpy as np
import trimesh
from scipy.ndimage import binary_erosion


def load_clean(path: str) -> trimesh.Trimesh:
    m = trimesh.load(path, process=True)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.remove_infinite_values()
    return m


def solidify_offset(mesh: trimesh.Trimesh, wall_mm: float = 2.0, smooth_iters: int = 0):
    """オープンシェルを法線方向に内側オフセットして一定肉厚の中空殻にする（solidify）。
    外面（歯列ディテール）を保持し、内面を wall_mm 内側に作成、開口縁を側壁で閉じる。
    結果はwatertightな肉厚一定の中空モデル（排出口は後工程で別途）。"""
    if smooth_iters > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=int(smooth_iters))
    V = mesh.vertices
    N = mesh.vertex_normals
    n = len(V)
    inner = V - N * float(wall_mm)            # 内側オフセット面
    verts = np.vstack([V, inner])
    outer_f = mesh.faces                      # 外面（向きそのまま）
    inner_f = mesh.faces[:, ::-1] + n         # 内面（反転＋index offset）
    # 開口縁（境界エッジ）を側壁で連結
    sorted_e = [tuple(sorted(e)) for e in mesh.edges]
    cnt = Counter(sorted_e)
    walls = []
    for a, b in mesh.edges:                    # 面順の有向エッジ
        if cnt[tuple(sorted((a, b)))] == 1:    # 境界エッジ
            walls.append([a, b, b + n])
            walls.append([a, b + n, a + n])
    faces = np.vstack([outer_f, inner_f, np.array(walls, dtype=np.int64)]) if walls \
        else np.vstack([outer_f, inner_f])
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    out.merge_vertices()
    trimesh.repair.fix_normals(out)
    info = dict(method="offset", wall_mm=wall_mm, boundary_walls=len(walls) // 2)
    return out, info


def _boundary_vertex_mask(mesh):
    sorted_e = [tuple(sorted(e)) for e in mesh.edges]
    cnt = Counter(sorted_e)
    bmask = np.zeros(len(mesh.vertices), bool)
    bnd_edges = []
    for a, b in mesh.edges:
        if cnt[tuple(sorted((a, b)))] == 1:
            bmask[a] = bmask[b] = True
            bnd_edges.append((a, b))
    return bmask, bnd_edges


def smooth_boundary(mesh, iters=12, lam=0.6):
    """境界（辺縁）頂点のみをラプラシアン平滑（歯面ディテールは触らない）。"""
    from scipy.spatial import cKDTree
    V = mesh.vertices.copy()
    bmask, _ = _boundary_vertex_mask(mesh)
    if not bmask.any():
        return mesh
    bidx = np.where(bmask)[0]
    tree = cKDTree(V[bidx])
    k = min(8, len(bidx))
    _, nn = tree.query(V[bidx], k=k)
    for _ in range(iters):
        avg = V[bidx][nn].mean(axis=1)
        V[bidx] = (1 - lam) * V[bidx] + lam * avg
    m = mesh.copy(); m.vertices = V
    return m


def solidify_surface_preserving(mesh: trimesh.Trimesh, wall_mm: float = 1.2,
                                smooth_boundary_iters: int = 12, open_base: bool = True,
                                base_axis: int = 2):
    """表面保持の肉厚付け→pymeshfixで修復しwatertight薄肉殻。
    open_base=Trueなら歯肉側（base_axis下端）を平面カットして開放底にする。"""
    import pymeshfix
    m = smooth_boundary(mesh, iters=smooth_boundary_iters)
    V = np.asarray(m.vertices, np.float64)
    N = np.asarray(m.vertex_normals, np.float64)
    n = len(V)
    inner = V - N * float(wall_mm)
    verts = np.vstack([V, inner])
    _, bnd = _boundary_vertex_mask(m)
    walls = []
    for a, b in bnd:
        walls += [[a, b, b + n], [a, b + n, a + n]]
    faces = np.vstack([m.faces, m.faces[:, ::-1] + n, np.array(walls, np.int64)])
    shell = trimesh.Trimesh(verts, faces, process=True)
    # 自己交差・非多様体を修復してwatertight化（表面はリメッシュせず保持寄り）
    vin = np.ascontiguousarray(shell.vertices, dtype=np.float64)
    fin = np.ascontiguousarray(shell.faces, dtype=np.int32)
    vc, fc = pymeshfix.clean_from_arrays(vin, fin)
    out = trimesh.Trimesh(vc, fc, process=True)
    info = dict(method="offset+meshfix", wall_mm=wall_mm,
                watertight=bool(out.is_watertight), volume=float(out.volume) if out.is_volume else None)
    if open_base and out.is_watertight:
        # 基底面は「歯肉側」に置く（切縁/咬合側の歯を削らない）。
        # 歯肉側 = tooth-only入力の開口境界(歯頸ライン)がある側。
        bmask, _ = _boundary_vertex_mask(mesh)   # 平滑前の元meshで歯頸境界を取得
        gc = np.asarray(mesh.vertices).mean(axis=0)
        bc = np.asarray(mesh.vertices)[bmask].mean(axis=0) if bmask.any() else gc + np.array([0, 0, 1.0])
        gum_dir = bc - gc
        nrm = np.linalg.norm(gum_dir)
        gum_dir = gum_dir / nrm if nrm > 1e-6 else np.array([0.0, 0.0, 1.0])
        proj = np.asarray(out.vertices) @ gum_dir
        cut_at = float(proj.max() - wall_mm * 2.0)     # 歯肉最端から2mm内側で平らに
        origin = (gum_dir * cut_at).tolist()
        out = out.slice_plane(plane_origin=origin, plane_normal=(-gum_dir).tolist())  # 歯側を残す
        info["gum_dir"] = [round(float(x), 2) for x in gum_dir]
        info["open_base_cut"] = cut_at
    return out, info


def occlusal_normal(vertices, boundary_mask):
    """咬合平面の法線をPCA最小分散軸で推定し、歯肉側を向く符号に揃えて返す。
    歯列アーチは板状なので最小固有ベクトル≈咬合平面法線（=模型の上下軸）。"""
    V = np.asarray(vertices, np.float64)
    C = np.cov(V.T)
    w, vv = np.linalg.eigh(C)
    nrm = vv[:, 0]                                  # 最小固有値の軸
    gc = V.mean(axis=0)
    bc = V[boundary_mask].mean(axis=0) if boundary_mask.any() else gc + nrm
    if (bc - gc) @ nrm < 0:                         # 歯肉側を向くように符号調整
        nrm = -nrm
    return nrm / (np.linalg.norm(nrm) + 1e-9)


def _all_boundary_loops(bnd):
    """境界エッジ集合から全頂点ループ（順序付き）を長い順で返す。"""
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in bnd:
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    seen = set(); loops = []
    for s in list(adj):
        if s in seen:
            continue
        loop = [s]; seen.add(s); cur = s; prev = None
        while True:
            nxt = None
            for x in adj[cur]:
                if x == prev:
                    continue
                if x == s and len(loop) > 2:
                    nxt = s; break
                if x not in seen:
                    nxt = x; break
            if nxt is None or nxt == s:
                break
            loop.append(nxt); seen.add(nxt); prev = cur; cur = nxt
        loops.append(loop)
    return sorted(loops, key=len, reverse=True)


def _order_boundary_loop(bnd):
    """境界エッジ集合から最長の頂点ループ（順序付き）を返す。"""
    return _all_boundary_loops(bnd)[0]


def _fill_small_holes(m):
    """最大ループ（外周リム）以外の内部穴を平面三角形分割で塞ぐ。リムは開けたまま。"""
    _, bnd = _boundary_vertex_mask(m)
    loops = _all_boundary_loops(bnd)
    if len(loops) <= 1:
        return m
    V = np.asarray(m.vertices, np.float64)
    new_faces = list(m.faces)
    for loop in loops[1:]:                       # 最大=リムは残し、他を塞ぐ
        pts = V[loop]
        c = pts.mean(axis=0)                     # 法線推定→平面基底
        _, _, vt = np.linalg.svd(pts - c)
        u, w = vt[0], vt[1]
        coords = np.column_stack([(pts - c) @ u, (pts - c) @ w])
        try:
            from shapely.geometry import Polygon
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            _, f2 = trimesh.creation.triangulate_polygon(poly, engine="earcut")
            for tri in f2:
                new_faces.append([loop[tri[0]], loop[tri[1]], loop[tri[2]]])
        except Exception:                        # 失敗時はファン三角形
            for i in range(1, len(loop) - 1):
                new_faces.append([loop[0], loop[i], loop[i + 1]])
    out = trimesh.Trimesh(V, np.asarray(new_faces, np.int64), process=True)
    out.merge_vertices()
    return out


def extrude_base(mesh: trimesh.Trimesh, depth: float = 10.0, base_dir=None,
                 smooth_boundary_iters: int = 8):
    """咬合平面に垂直な方向（=模型の上下軸）に開口縁を押し出し、咬合平面に平行な
    平底で閉じて中実の模型ベースにする。底面はU字境界ループを拘束三角形分割するため、
    口蓋開口部を橋渡しする切れ込み・多面が出ない。表面（歯列）は保持。"""
    m = smooth_boundary(mesh, iters=smooth_boundary_iters) if smooth_boundary_iters else mesh.copy()
    m = _fill_small_holes(m)                    # 外周リム以外の内部穴を塞ぐ（watertight化）
    bmask, bnd = _boundary_vertex_mask(m)
    V = np.asarray(m.vertices, np.float64)
    bd = occlusal_normal(V, bmask) if base_dir is None else \
        np.asarray(base_dir, float) / (np.linalg.norm(base_dir) + 1e-9)
    loop = _order_boundary_loop(bnd)                      # 順序付き境界ループ
    L = len(loop)
    base_level = float((V @ bd).max()) + float(depth)
    # base_dirに直交する平面基底(U,Vv)を作り、ループを2Dへ投影
    a = np.array([1.0, 0, 0]) if abs(bd[0]) < 0.9 else np.array([0, 1.0, 0])
    U = np.cross(bd, a); U /= np.linalg.norm(U)
    Vv = np.cross(bd, U)
    P = V[loop]
    coords2d = np.column_stack([P @ U, P @ Vv])
    from shapely.geometry import Polygon
    poly = Polygon(coords2d)
    if not poly.is_valid:
        poly = poly.buffer(0)
    cap_v2d, cap_f = trimesh.creation.triangulate_polygon(poly, engine="earcut")
    # 底リング（ループをbase平面へ投影）をL頂点で明示生成し、earcut面を座標照合で
    # リング頂点へリマップ（earcutの頂点数差・閉点重複に依存しない頑健化）
    ring = P - np.outer(P @ bd - base_level, bd)
    from scipy.spatial import cKDTree
    _, cap_to_ring = cKDTree(coords2d).query(cap_v2d)
    n = len(V)
    allV = np.vstack([V, ring])
    cap_faces = cap_to_ring[np.asarray(cap_f, np.int64)] + n
    walls = []                                            # 側壁（上ループ↔底リング）
    for i in range(L):
        a0 = loop[i]; b0 = loop[(i + 1) % L]
        a1 = n + i; b1 = n + (i + 1) % L
        walls += [[a0, b0, b1], [a0, b1, a1]]
    faces = [m.faces, cap_faces, np.array(walls, np.int64)]
    out = trimesh.Trimesh(allV, np.vstack(faces), process=True)
    out.merge_vertices()
    out.update_faces(out.unique_faces() & out.nondegenerate_faces())
    # 残る微小穴（歯表面のスキャン欠損など）を全て塞ぐ
    for _ in range(3):
        if out.is_watertight:
            break
        trimesh.repair.fill_holes(out)
        _, rb = _boundary_vertex_mask(out)
        if not rb:
            break
        Vc = np.asarray(out.vertices, np.float64); nf = list(out.faces)
        for lp in _all_boundary_loops(rb):       # 残ループをファンで閉じる
            for i in range(1, len(lp) - 1):
                nf.append([lp[0], lp[i], lp[i + 1]])
        out = trimesh.Trimesh(Vc, np.asarray(nf, np.int64), process=True)
        out.merge_vertices()
    trimesh.repair.fix_normals(out)
    info = dict(method="extrude_base", depth=depth, base_dir=[round(float(x), 2) for x in bd],
                loop_len=L, watertight=bool(out.is_watertight),
                volume=float(out.volume) if out.is_volume else None)
    return out, info


def _to_volume(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """pymeshfixで確実にwatertightな体積メッシュ（is_volume）に整える。"""
    import pymeshfix
    vc, fc = pymeshfix.clean_from_arrays(
        np.ascontiguousarray(mesh.vertices, np.float64),
        np.ascontiguousarray(mesh.faces, np.int32))
    out = trimesh.Trimesh(vc, fc, process=True)
    trimesh.repair.fix_normals(out)
    return out


def make_hollow_open_model(mesh: trimesh.Trimesh, rim_mm: float = 3.0,
                           wall_mm: float = 2.0, pitch: float = 0.2,
                           cavity_smooth: int = 8, open_bottom: bool = False,
                           base_dir=None):
    """歯列ディテールを完全保持したまま中空化する（ブーリアン差分方式）。
    外殻=歯列＋rim_mm土台リムの中実ソリッド（pymeshfixでvolume化、表面はそのまま）。
    内部キャビティ=外殻を粗ボクセル(pitch)で wall_mm 収縮した滑らかなソリッド。
    hollow = 外殻 − キャビティ。外形は厳密に保持され、咬合面スパイクは原理的に出ず、
    結果はwatertight（既定は底もリムも閉じた中空）。open_bottom=Trueなら底側を
    ボックス差分して開口（断面はブーリアンで閉じる＝リム肉厚は保持）。"""
    from scipy.ndimage import binary_erosion, binary_fill_holes
    # 1) 詳細外殻ソリッド（歯列＋rim_mmリム＋平底）→ volume化
    solid, si = extrude_base(mesh, depth=rim_mm, base_dir=base_dir)
    bd = np.asarray(si["base_dir"], np.float64)
    bd = bd / (np.linalg.norm(bd) + 1e-9)
    solid = _to_volume(solid)
    # 2) 内部キャビティ：粗ボクセルで内側へ wall_mm 収縮した滑らかソリッド
    vg = solid.voxelized(pitch=float(pitch))
    filled = binary_fill_holes(np.asarray(vg.matrix, dtype=bool))
    n = max(1, int(round(float(wall_mm) / float(pitch))))
    cavity_vox = binary_erosion(filled, iterations=n)
    cavity = trimesh.voxel.ops.matrix_to_marching_cubes(cavity_vox, pitch=float(pitch))
    cavity.apply_translation(vg.transform[:3, 3])
    if cavity_smooth > 0:
        trimesh.smoothing.filter_taubin(cavity, iterations=int(cavity_smooth))
    cavity = _to_volume(cavity)
    # 3) ブーリアン差分＝中空（外形保持・watertight）
    hollow = trimesh.boolean.difference([solid, cavity])
    info = dict(method="hollow_boolean", rim_mm=rim_mm, wall_mm=wall_mm, pitch=pitch,
                base_dir=[round(float(x), 2) for x in bd])
    # 4) open_bottom：底側をボックス差分して開口（リム断面はブーリアンで閉鎖）
    if open_bottom:
        base_level = float((np.asarray(solid.vertices) @ bd).max())
        cut = base_level - max(float(wall_mm) * 1.3, 1.0)
        ext = float(np.ptp(np.asarray(solid.vertices), axis=0).max()) * 2.0
        box = trimesh.creation.box(extents=[ext, ext, ext])
        # ボックスをbd+側（底より外）へ寄せ、cut面までを覆う
        box.apply_translation((bd * (cut + ext / 2.0)).tolist())
        hollow = trimesh.boolean.difference([hollow, _to_volume(box)])
        info["open_cut_level"] = round(cut, 2)
    info["watertight"] = bool(hollow.is_watertight)
    info["faces"] = int(len(hollow.faces))
    return hollow, info


def voxel_hollow(mesh: trimesh.Trimesh, pitch: float = 0.3, wall_mm: float = 2.0,
                 smooth_iters: int = 8):
    """ボクセル厚み付け方式（オープンシェル向け・頑健）。
    サーフェスをボクセル化→dilationで一定肉厚に膨らませ→marching cubesでメッシュ化。
    自己交差せず必ずwatertight。外形は歯列サーフェスに追従、ドーム下面は開放のまま。"""
    from scipy.ndimage import binary_dilation
    vg = mesh.voxelized(pitch=pitch)
    solid = np.asarray(vg.matrix, dtype=bool)
    n = max(1, int(round(wall_mm / (2.0 * pitch))))   # 両側に膨らむので総厚≈(2n+1)*pitch
    thick = binary_dilation(solid, iterations=n)
    out = trimesh.voxel.ops.matrix_to_marching_cubes(thick, pitch=pitch)
    out.apply_translation(vg.transform[:3, 3])        # 原座標へ
    if smooth_iters > 0:
        trimesh.smoothing.filter_taubin(out, iterations=int(smooth_iters))
    approx_wall = round((2 * n + 1) * pitch, 2)
    return out, dict(method="voxel", pitch=pitch, approx_wall_mm=approx_wall,
                     surf_voxels=int(solid.sum()), thick_voxels=int(thick.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--pitch", type=float, default=0.35)
    ap.add_argument("--wall", type=float, default=2.0)
    ap.add_argument("--smooth", type=int, default=8)
    ap.add_argument("--method", choices=["offset", "voxel"], default="voxel")
    a = ap.parse_args()

    m = load_clean(a.inp)
    if a.method == "offset":
        out, info = solidify_offset(m, a.wall, a.smooth)
    else:
        out, info = voxel_hollow(m, a.pitch, a.wall, a.smooth)
    out.export(a.out)
    ext = out.bounds[1] - out.bounds[0]
    print("INPUT :", len(m.vertices), "v /", len(m.faces), "f  watertight=", m.is_watertight)
    print("OUTPUT:", len(out.vertices), "v /", len(out.faces), "f  watertight=", out.is_watertight)
    print("bbox(mm):", [round(float(x), 1) for x in ext])
    print("info:", info)
    print("saved:", a.out)


if __name__ == "__main__":
    main()
