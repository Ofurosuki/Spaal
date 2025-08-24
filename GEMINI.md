# About this project
このプロジェクトは、複数機種のLiDARについて照射したレーザーとその反射波の波形を各スキャンについて保存し、受信するsimulatorである。
それに加え、HFR攻撃(High Frequency Attack)と呼ばれる高周波のレーザを外部から照射し、ジャミング攻撃する装置のsimulatorもあり、(spoofer)
ジャミングを受けた信号についてreconstructを試みるモジュールが存在する。

Hist-matrixとは、 altitude x Azimuth x Histdata の形式でLiDARのHistgramを表現する行列。hist-data本体、入力の.pcdファイルの先頭行の点群(=LiDARの最初のスキャンとなる点)の世界座標に対する水平Offset角度、LiDARの垂直角の方向のリスト、LiDARのFoV(360°LiDARなら360)、時間分解能(ns)を含み、これを.npzとして出力する。これが標準的なデータのフォーマットになっている。

# Project Structure
- /spaal2 : 基本的なLiDARのsimulator。
- /pcd_datasets : hist-matrixの.npzファイルを保管する。
- /reconstruction : HFR攻撃によるジャミングから信号を復元するモジュール。
- /evaluation : HFR攻撃からどれくらい信号を復元できたのかを評価するコードがある。

# 実行環境
uvを使用しているので、uv run ~~とする。

# coding rules
- commentは英語で書く。
- 変数や関数定義時に型を記述する。

# 開発上の注意点 (Development Notes)

## codeの変更時
変更時は何を変更するのか概要を提示して、実行の可否を訪ねてから変更を実行してください。

### `hist-matrix` (.npz) の仕様
- **ソース**: 複数のPCDファイル（ディレクトリ単位で指定）を元に、複数フレームを持つ単一の`.npz`ファイルを生成する。
- **方位角オフセット**:
    - `.npz`ファイルには、フレームごとの初期方位角オフセットが `initial_azimuth_offsets` というキーのNumpy配列で格納されている。
    - これは、各フレームの元となったPCDファイルごとに計算されたオフセットのリストである。

### `hist_matrix_visualizer.py` における方位角の再構成
点群を再構成する際の方位角の計算は、以下の式で行うことが必須である。

```python
# in datasets_generator/hist_matrix_visualizer.py
azimuth_deg = (h_idx / horizontal_resolution) * self.fov + current_azimuth_offset
```

**重要**: `current_azimuth_offset` を **加算(`+`)** するのがこのプロジェクトのパイプラインにおける正しい仕様である。一見、逆変換のために減算(`-`)するように見えるが、それは誤りであり、点群のずれを引き起こす。

