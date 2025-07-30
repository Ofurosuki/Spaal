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