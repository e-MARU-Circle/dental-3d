# Model Segmentator（推論最小セット）— dental-3d パイプライン用

歯科3Dパイプライン（dental-3d / blender-dental スキルの run_pipeline.py）が
歯/歯肉セグメンテーションに使う **推論専用の最小セット** です。
学習用データ・学習スクリプト・大容量チェックポイントは含みません。
本モデルは一般公開データセットを用いて著者が学習・作成したものです（利用元データセットのライセンスに従ってください）。

## 含まれるもの
- `ml/`         … 推論に必要なコード（models / data / infer / utils）
- `ckpts/stage1_last.pth`        … 歯/歯肉分割の学習済み重み（stage1・本パイプラインで使用）
- `ckpts/stage2_learned_last.pth`… 参考（インスタンス分離用・本パイプラインでは未使用）
- `pyproject.toml` / `requirements.txt` / `configs/`

## 使い方
1. このフォルダを任意の場所へ展開（例: `~/ModelSegmentator_min`）。
2. パイプライン側の依存を導入（dental-3d の pipeline/requirements.txt）。
3. 実行時に `--repo` でこのフォルダを指定:

```
python3 run_pipeline.py --in SCAN.stl --out model.stl --arch upper \
  --repo /path/to/ModelSegmentator_min
```

## 注意
- 重みは学習時フレーム前提。新規スキャンの座標正準化（rotX180→XY中心化→上端Z=-78）は
  パイプライン側で自動処理されます。
- 患者氏名・IDをファイル名／出力に含めないこと。
