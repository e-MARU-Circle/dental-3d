# 中空オープン歯科模型 パイプライン

口腔内スキャン（STL）から、3Dプリント用の **中空・オープン底の歯科模型** を1コマンドで生成する。

## クイックスタート

```bash
pip install -r requirements.txt
python3 run_pipeline.py \
  --in  SCAN_upper.stl \
  --out model_upper.stl \
  --arch upper \
  --repo /path/to/Model\ Segmentator
```

出力 `model_upper.stl` は watertight な中空オープン模型。スライサー（PrusaSlicer / Cura）にそのまま投入できる。

## 処理工程

| 順 | 処理 | 関数 |
|----|------|------|
| 1 | スキャン読込・正準化・歯/歯肉ML分割 | `ml_segment.segment` |
| 2 | 歯＋歯肉バンド(5mm)抽出 | `ml_segment.extract_with_gum_band` |
| 3 | 口蓋カットを放物線フィットで平滑化（歯は無条件保持） | `ml_segment.smooth_palatal_cut` |
| 4 | メッシュ簡略化 | `fast_simplification` |
| 5 | 咬合平面(PCA最小軸)に垂直な土台リム延長＋平底 | `geometry_ops.extrude_base` |
| 6 | ブーリアン中空化（外形保持・自己交差なし）＋底開放 | `geometry_ops.make_hollow_open_model` |

## オプション

| フラグ | 既定 | 説明 |
|--------|------|------|
| `--arch` | upper | `upper` / `lower` |
| `--rim` | 3.0 | 土台リム高さ(mm) |
| `--wall` | 2.0 | 肉厚(mm) |
| `--pitch` | 0.2 | 内部キャビティ解像度(mm)。小さいほど内面が滑らか・処理重 |
| `--keep-pct` | 40 | 口蓋カラー幅（大きいほど口蓋ガムを広く残す） |
| `--closed` | — | 底を閉じた中空にする |
| `--solid` | — | 中空にせず中実土台にする |
| `--repo` | Mac実パス | Model Segmentator の場所 |

確定パラメータ（2026-06-22 承認）: rim 3 / wall 2 / pitch 0.2 / keep_pct 40。

## 設計上のキモ

- **咬合平面の認識**: 全頂点のPCA最小分散軸を咬合平面法線とし、その方向に土台を垂直押し出し。歯肉重心方向だと斜めに尖るため不可。
- **口蓋カットの平滑化**: 現状の口蓋縁点群に放物線(degree=4)を最小二乗フィットし、その内側オフセット曲線で切り直す。歯頂点を含む面＋歯から0.8mm以内は無条件保持で **歯は絶対に削らない**。
- **中空化はブーリアン差分方式**: 詳細外殻ソリッドをpymeshfixで体積化し、粗ボクセル(pitch)で肉厚分収縮した内部キャビティを作って `solid − cavity`。これで外形ディテール完全保持・咬合面スパイクなし・watertight・リム断面閉鎖を同時達成。
  - 法線内側オフセット方式は臼歯溝で自己交差しスパイクが出るため不可。
  - 純ボクセル収縮方式は表面が階段状に荒れ形態が崩れるため不可。

## 前提・トラブルシュート

- **Model Segmentator が必要**: 歯/歯肉分割の学習済みモデル本体。本パッケージには同梱しない。`--repo` で場所を指定。新規スキャンは学習フレームへの正準化（rotX180 → XY中心化 → 上端Z=-78）が必須（未実施だと歯認識率ほぼ0%）。
- **メモリ**: ML(torch)とボクセル化を同時に走らせるとRAMを多く使う。低メモリ環境では工程分割を推奨。
- **Blenderは不要**: 本パイプライン自体はBlender非依存。確認レンダリングのみBlender/MCPを使う（`../scripts/render_check.py`）。
- **患者情報**: STLファイル名・出力に患者氏名/IDを含めない（院内ハードルール準拠）。

## 依存

`requirements.txt` 参照。主要: trimesh / scipy / shapely / scikit-image / fast-simplification / pymeshfix / manifold3d / mapbox-earcut / torch / scikit-learn。
