"""
歯冠分離＝歯根除去 (blender-dental: ワークフローB)

歯根付きモデルから、CEJ（歯頸線）相当の水平面より下を切り落として歯冠のみを残す。
非破壊: 元オブジェクトは複製し、複製側を "<name>_crown" として加工する。

使い方:
  isolate_crown("tooth_11", cej_z=None)   # cej_z未指定ならバウンディングボックスから推定
推定ロジック: オブジェクト高さの下から CEJ_RATIO の位置を歯頸線とみなす（歯種で要調整）。

実行: Blender MCP の execute_blender_code に本ファイル＋呼び出し行を渡す。
注意: ハードルール5（研究データを破壊的に変換しない）。必ず複製して別名保存。
"""
import bpy
import bmesh
from mathutils import Vector

CEJ_RATIO = 0.45  # 下端から見た歯頸線のおおよその高さ比。前歯は低め/臼歯は高めに調整。


def _bbox_world_z(obj):
    zs = [(obj.matrix_world @ Vector(c)).z for c in obj.bound_box]
    return min(zs), max(zs)


def isolate_crown(obj_name, cej_z=None, ratio=CEJ_RATIO):
    src = bpy.data.objects.get(obj_name)
    if src is None:
        print(f"[ERR] object not found: {obj_name}")
        return None

    # 非破壊: 複製
    dup = src.copy()
    dup.data = src.data.copy()
    dup.name = f"{obj_name}_crown"
    for c in src.users_collection:
        c.objects.link(dup)

    zmin, zmax = _bbox_world_z(dup)
    if cej_z is None:
        cej_z = zmin + (zmax - zmin) * ratio

    # bisect で水平面カットし、下側（歯根側）を除去 → 開口部を埋める
    bpy.context.view_layer.objects.active = dup
    bpy.ops.object.select_all(action='DESELECT')
    dup.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(dup.data)
    bmesh.ops.bisect_plane(
        bm,
        geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
        plane_co=(0, 0, cej_z),
        plane_no=(0, 0, 1),
        clear_inner=True,   # 下側(歯根)を削除
    )
    bmesh.update_edit_mesh(dup.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.fill_holes(sides=0)  # 切断面を閉じる
    bpy.ops.object.mode_set(mode='OBJECT')
    dup["crown_only"] = True
    print(f"[DONE] {dup.name} crown isolated at z={cej_z:.3f}")
    return dup


# 例: isolate_crown("tooth_11")
