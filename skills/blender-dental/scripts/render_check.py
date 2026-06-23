"""
レンダリング確認 (blender-dental: ワークフローD / 検証ステップ)

Teeth_Library 全体が収まるようカメラ・ライトを自動配置し、PNGを出力する。
作業後の目視検証に使用（このスキルの必須仕上げ）。

実行: Blender MCP の execute_blender_code に本ファイルを渡す。
出力先 OUT_PATH は先生のローカルパス。必要に応じて変更。
"""
import bpy
from mathutils import Vector

OUT_PATH = "/Users/ema/Desktop/VScode/PenClaw/assets/3d_library/teeth/_render_check.png"
COLLECTION_NAME = "Teeth_Library"


def _scene_bounds():
    col = bpy.data.collections.get(COLLECTION_NAME)
    objs = list(col.objects) if col else [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not objs:
        return Vector((0, 0, 0)), 1.0
    mins = Vector((1e9, 1e9, 1e9)); maxs = Vector((-1e9, -1e9, -1e9))
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            mins = Vector((min(mins[i], w[i]) for i in range(3)))
            maxs = Vector((max(maxs[i], w[i]) for i in range(3)))
    center = (mins + maxs) / 2
    radius = max((maxs - mins).length / 2, 1.0)
    return center, radius


def render_check(out_path=OUT_PATH):
    center, radius = _scene_bounds()

    cam_data = bpy.data.cameras.new("CheckCam")
    cam = bpy.data.objects.new("CheckCam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = center + Vector((0, -radius * 3.0, radius * 2.0))
    # カメラを中心に向ける
    direction = (center - cam.location)
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam

    light_data = bpy.data.lights.new("CheckSun", type='SUN')
    light = bpy.data.objects.new("CheckSun", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = center + Vector((radius, -radius, radius * 3))

    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = out_path
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960
    bpy.ops.render.render(write_still=True)
    print(f"[DONE] rendered -> {out_path}")


render_check()
