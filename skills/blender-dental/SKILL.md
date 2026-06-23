---
name: blender-dental
description: "Blender×歯科3Dワークフロー専用スキル。(1)口腔内スキャン(STL)から3Dプリント用の中空オープン模型を1コマンド生成するパイプライン（歯/歯肉ML分割→口蓋カット放物線平滑化→咬合平面に垂直な土台→ブーリアン中空化→底開放）。(2)歯牙3Dライブラリの読み込み・FDI命名整列・歯冠のみ化・レンダリング確認。「中空模型」「オープン模型」「3Dプリント模型」「歯型STL」「模型を作って」「中空化」「土台をつけて」「口蓋カット」「run_pipeline」「歯牙ライブラリ」「歯冠のみ」「歯根を外す」「FDI整列」「3D歯科ワークフロー」と言われたら発動。実機操作はBlender MCP（コード=penclaw-ml）と連携。"
---

# blender-dental — Blender歯科3Dワークフロー

歯牙3Dライブラリを使ったBlender作業を定型化する補完スキル。3D実行の主担当は**コード（penclaw-ml）**、本スキルは「ライブラリ運用の手順とスクリプト」を提供する。

## 前提

- Blenderが起動し、MCPアドオンのサーバーが `localhost:9876` で稼働していること（未起動だと全操作が接続エラー）。
- ライブラリ実体: `assets/3d_library/teeth/`（README/SOURCES/manifest/models）。命名規則は `tooth_<FDI2桁>_<英名>_crown.<ext>`。
- スクリプト: 本スキルの `scripts/` 配下。Blender MCP の `execute_blender_code` から実行する。

## 起動時チェック（必ず最初に）

1. `get_blendfile_summary_path_info` でBlender接続を確認。失敗したら**先生にBlender起動を依頼して中断**。
2. `get_objects_summary` で現在のシーン構成を把握（既存`Teeth_Library`コレクションの有無）。

## 主要ワークフロー

### ★ 中空オープン模型パイプライン（`pipeline/run_pipeline.py`）★
口腔内スキャン(STL)から3Dプリント用の中空オープン模型を一括生成する主力機能。
詳細・依存・トラブルシュートは `pipeline/README.md`、設計は `pipeline/PIPELINE_shell_to_hollow.md`。

```
python3 pipeline/run_pipeline.py --in SCAN.stl --out model.stl --arch upper
```

工程: スキャン読込 → 歯/歯肉ML分割(Model Segmentator stage1) → 歯＋歯肉バンド抽出
→ 口蓋カットを放物線フィットで平滑化 → 簡略化 → 咬合平面に垂直な土台リム延長
→ ブーリアン中空化（外形保持・自己交差なし・watertight）→ 底開放 → 書出し。

確定パラメータ(2026-06-22承認): rim 3mm / wall 2mm / pitch 0.2 / keep_pct 40。
オプション: `--closed`(底閉じ中空) / `--solid`(中実土台) / `--arch lower`。

主要モジュール:
- `ml_segment.py`: `segment`(歯確率), `extract_with_gum_band`(歯＋歯肉5mm), `smooth_palatal_cut`(口蓋カット放物線平滑化・歯は無条件保持)
- `geometry_ops.py`: `occlusal_normal`(咬合平面=PCA最小軸), `extrude_base`(垂直土台＋平底), `make_hollow_open_model`(ブーリアン中空化)

前提: 別途 Model Segmentator リポジトリ＋学習済み重み（`--repo`で指定）と、`pipeline/requirements.txt` の依存が必要。Blenderは本パイプライン自体には不要（確認レンダリング時のみ）。

### A. ライブラリ読み込み（`scripts/import_tooth_library.py`）
`models/upper|lower` を走査し、stl/obj/glb/ply を1ファイル=1オブジェクトでインポート。
ファイル名のFDIで `tooth_11` 等にリネーム、`["fdi"]` カスタムプロパティを付与、`Teeth_Library` コレクションに格納。

### B. 歯冠のみ化＝歯根分離（`scripts/isolate_crown.py`）
歯根付きモデルしか入手できない場合に使用。CEJ（歯頸線）相当の水平面で下方をブーリアン/平面カットし歯冠のみを残す。カット高さは歯種ごとに引数指定。**元データは保持し、別名 `_crown` で出力**（破壊的変更を避ける）。

### C. FDI整列（`scripts/organize_fdi.py`）
上下顎弓の標準配置（または点検用グリッド）に各歯を並べる。デモ・患者説明・教育用の歯列セットアップに使用。

### D. レンダリング確認（`scripts/render_check.py`）
カメラ・ライトをセットし、配置結果をPNG出力して目視確認。作業後の検証ステップとして必須。

## 運用ルール（CLAUDE.md準拠）

- 既存シーン・命名を尊重。破壊的変更前に確認（Blender MCPの注意書きと整合）。
- 研究データ（STL/DICOM）は勝手に変換・削除しない（ハードルール5）。歯冠分離は必ず別名保存。
- モデル追加時は `manifest.csv` の source/license を更新。ライセンス不明品は使わない。
- 患者氏名・患者ID・カルテ情報をライブラリやシーンに含めない（ハードルール6）。

## 連携

- 実機3D操作・モデル評価: コード（penclaw-ml）
- スキル改修・配布・LFS設定: ハブ（penclaw-hub）
- 患者説明用の見せ方・薬機法チェック: ナナ（penclaw-patient-content）
