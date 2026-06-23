# 中空模型パイプライン設計（オープンシェルSTL → 中空模型）

目的: 口腔内スキャン等の**オープンシェルSTL**を、穴補修→ML境界スムーズ化→押し出し→中空化して、3Dプリント用の**中空模型ベース**に自動加工する。

決定事項（2026-06-20）: 用途=模型ベース（中空模型）／MLは先生のPointNet2研究モデル流用／実行=ハイブリッド（重処理Python・最終確認Blender）。担当: コード（ML＋3D）＋ハブ（パイプライン）、評価=ケン、臨床用途化時の薬機チェック=ナナ。

## 処理フロー（5工程）

| # | 工程 | 手法 | 基盤 | 確実性 |
|---|------|------|------|--------|
| 1 | 穴補修 | pymeshfix で非多様体修正＋穴埋め。境界ループ検出→trianglulate | Python | 高 |
| 2 | ML境界認識 | PointNet2で歯/歯肉セグメンテーション→歯列ラベル取得 | Python(PyTorch) | 中（モデル依存） |
| 3 | 辺縁スムーズ化 | 歯肉・辺縁領域のみ選択し境界を外側へ拡張→Taubin/Laplacianスムージング（歯冠は形態保持） | Python(Open3D) | 中 |
| 4 | 押し出し | 法線方向オフセットで厚み付け（solidify相当） | Python/Blender | 高 |
| 5 | 中空化 | 内側オフセットサーフェス生成→ブーリアン差、または均一肉厚シェル化＋排出口 | Python/Blender | 高 |
| 検証 | 目視確認 | Blenderで読込・断面・レンダリング | Blender(MCP) | — |

工程1・4・5・検証は枯れた処理で確実。**山は工程2-3**（MLで歯列を認識し、歯肉辺縁だけ選択的にスムージング）。

## I/O・パラメータ（暫定）

- 入力: オープンシェルSTL（単一）。単位mm前提。
- 出力: `<name>_hollow.stl`（中空模型）。中間物 `_filled` `_smoothed` `_solid` も保持（非破壊）。
- 主パラメータ: 肉厚(wall_mm, 既定2.0)、スムージング反復(iters)、境界拡張幅(dilation)、排出口径。

## ハイブリッド実行構成

```
pipeline/
├── PIPELINE_shell_to_hollow.md   ← この設計
├── geometry_ops.py   ← 工程1/4/5（trimesh+pymeshfix+Open3D）
├── ml_segment.py     ← 工程2 PointNet2ラッパ（研究モデルI/Oを差し込む）
├── smooth_margin.py  ← 工程3 選択的スムージング
└── run_pipeline.py   ← 一括実行(CLI)。重処理後、Blenderで最終確認に渡す
```
重い処理（ML・メッシュ演算）はヘッドレスPython、最終確認・微調整はBlender(bpy/MCP)。

## 研究モデル統合（所在・I/O確認済み 2026-06-20）

リポジトリ: `/Users/ema/Desktop/VScode/Model Segmentator`（PointNet2 Lite、2段階）。
- 推論エントリ: `ml/infer/segment_and_instance.py`（config＋ckpt指定でセマンティック＝歯/歯肉＋インスタンス＝FDIラベルを出力、TTA対応）
- 重み: `ckpts/stage1_last.pth`(987K) ほか `releases/`。**Lite（軽量）なのでCPU推論可**。
- 前処理: `ml/data/preprocess_features.py`（メッシュ→点群＋特徴）。設定 `configs/stage1_pointnet2_lite.yaml`。
- 既存資産で工程3が前進: `ml/infer/smooth.py`（kNNラプラシアン平滑）、`ml/infer/boundary_refine.py`（境界リファイン）、`ml/infer/assign_fdi.py`。

統合の要点: モデル出力は**点群上のラベル**。工程3で原メッシュ頂点に適用するため **点群ラベル→原メッシュ頂点へkNN転写**（最近傍）する1段を挟む。歯肉/辺縁頂点を選択→境界を外側へdilation→Taubin/ラプラシアン平滑（歯冠は形態保持）。

## 着手の残ブロッカー

1. ~~PointNet2モデルの所在・I/O~~ → **解消**（Model Segmentatorで確認済み）。
2. **テスト用オープンシェルSTL（匿名・非患者）1〜数件** — 開発検証用。患者情報は本フォルダに置かない（CLAUDE.mdルール6）。これが揃えばM1即着手。
3. 実行環境: サンドボックスにtorch未導入（CPU版を入れれば軽量モデルゆえ推論可）。本番はMac側(MPS)も可。

## マイルストーン

- M1: 工程1/4/5の幾何処理プロトタイプ（ML抜き）をサンプルで動作確認
- M2: PointNet2統合（工程2）→歯列ラベル取得
- M3: 工程3の選択的スムージング実装→全工程結線
- M4: blender-dental統合＋検証レンダリング、スキル化
