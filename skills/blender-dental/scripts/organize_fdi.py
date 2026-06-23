"""
FDI整列 (blender-dental: ワークフローC)

Teeth_Library 内の歯を、点検用グリッド（既定）または簡易歯列弓に並べる。
デモ・患者説明・教育用の歯列セットアップに使用。

mode="grid": FDI昇順で格子配置（欠損が一目で分かる）
mode="arch": 上下顎を前後に分け、ざっくり弓状に配置（精密咬合ではない）

実行: Blender MCP の execute_blender_code に本ファイルを渡す。
"""
import bpy
import math

COLLECTION_NAME = "Teeth_Library"
SPACING = 12.0   # mm想定の配置間隔。モデルスケールに合わせて調整。


def _teeth():
    col = bpy.data.collections.get(COLLECTION_NAME)
    if not col:
        print(f"[ERR] collection '{COLLECTION_NAME}' not found")
        return []
    objs = [o for o in col.objects if o.get("fdi")]
    return sorted(objs, key=lambda o: o["fdi"])


def organize(mode="grid"):
    teeth = _teeth()
    if not teeth:
        return
    if mode == "grid":
        per_row = 8
        for i, o in enumerate(teeth):
            o.location = ((i % per_row) * SPACING, -(i // per_row) * SPACING, 0)
    elif mode == "arch":
        # 上顎(1x,2x)=後列 / 下顎(3x,4x)=前列。中心から左右へ展開する簡易弓。
        for o in teeth:
            fdi = o["fdi"]; q = int(fdi[0]); pos = int(fdi[1])
            row_y = SPACING * 2 if q in (1, 2) else 0
            sign = -1 if q in (1, 4) else 1
            x = sign * pos * SPACING
            y = row_y - (pos ** 2) * 0.3  # ゆるい弧
            o.location = (x, y, 0)
    print(f"[DONE] arranged {len(teeth)} teeth (mode={mode})")


# 例: organize("grid")
organize("grid")
