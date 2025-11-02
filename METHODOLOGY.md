## 実験セットアップとワークフロー (Experimental Setup and Workflow)

本研究で用いるシミュレーションパイプラインは、LiDAR（Light Detection and Ranging）センサの信号生成、高周波リフレクション（HFR）攻撃の注入、そして攻撃を受けた信号の復元という一連のプロセスを再現するために構築されている。本パイプラインは、再現可能な実験を保証するため、明確に定義されたコンポーネントとパラメータで構成される。

### 1. 全体ワークフロー (Overall Workflow)

実験のワークフローは以下の4つの主要なステージで構成される。

1.  **データ準備 (Data Preparation)**:
    *   入力として、KITTIデータセットなどで利用される`.bin`形式、または標準的な`.pcd`形式の3D点群データを使用する。
    *   これらの点群データを、本シミュレータの標準フォーマットである`hist-matrix` (`.npz`形式) に変換する。この`hist-matrix`は、LiDARの各スキャンにおけるレーザーの反射波形を、`[チャンネル, 方位角ステップ, 時間]`の3次元テンソルとして格納したものである。

2.  **LiDARシミュレーションとHFR攻撃 (LiDAR Simulation and HFR Attack)**:
    *   `hist-matrix`生成プロセス中に、指定されたLiDARモデル（例: Velodyne HDL-64E）の動作をシミュレートする。
    *   同時に、設定されたパラメータに基づきHFR攻撃信号を生成し、正当なLiDAR信号に重畳させる。これにより、攻撃を受けた状態の`hist-matrix`が生成される。

3.  **信号復元 (Signal Reconstruction)**:
    *   攻撃を受けた`hist-matrix`を入力とし、HFR攻撃の周波数を特定する。
    *   特定された周波数に基づき、攻撃信号成分を除去（フィルタリング）し、元の信号波形を復元する。復元された`hist-matrix`がこのステージの出力となる。

4.  **評価 (Evaluation)**:
    *   攻撃を受ける前の「正解データ」(`answer_matrix`)と、「復元されたデータ」を比較する。
    *   距離推定の平均絶対誤差（Mean Absolute Error, MAE）などの指標を用いて、復元アルゴリズムの精度を定量的に評価する。

---

### 2. コンポーネントとパラメータ (Components and Parameters)

#### 2.1. データ準備とシミュレーション (Data Preparation and Simulation)

このステージは `datasets_generator/hist_matrix_generator.py` スクリプトによって実行される。

*   **責務**:
    *   指定されたディレクトリ内の点群ファイル（`.pcd` or `.bin`）を読み込む。
    *   LiDARモデルの物理的特性（スキャンパターン、角度分解能など）をシミュレートする。
    *   HFR攻撃をシミュレートし、LiDAR信号に注入する。
    *   最終的な`hist-matrix`（信号、ラベル、正解データなどを含む）を`.npz`ファイルとして出力する。

*   **主要な実行時パラメータ**:
    *   `--lidar-type`: シミュレートするLiDARモデル。例: `PCD_HDL64E`, `PCD_VLP32c`。
    *   `--pcd-directory`: 入力点群ファイルが格納されているディレクトリのパス。
    *   `--output-dir`: 生成された`hist-matrix`を保存するディレクトリ。
    *   `--time-resolution-ns`: シミュレーションの時間分解能（単位:ナノ秒）。例: `1.0`。
    *   `--spoofer-type`: 実行する攻撃の種別。`adaptive_hfr_perturbation`（適応的HFR攻撃）または`off`。
    *   `--spoofer-angle`: 攻撃を開始するトリガーとなる方位角（deg）。LiDAR正面が0度。
    *   `--spoofer-altitude`: 攻撃を開始するトリガーとなる仰俯角（deg）。
    *   `--spoofer-width-deg`: 攻撃が影響を及ぼす方位角の範囲（deg）。
    *   `--horizontal-resolution-deg`: LiDARの内部的な水平方向の角度分解能。
    *   `--output-horizontal-resolution-deg`: 出力される`hist-matrix`の水平方向の解像度。

#### 2.2. LiDARモデル (LiDAR Models)

LiDARモデルは `spaal2/core/dummy_lidar/` 内で定義されている。`dummy_lidar_hdl64e.py` を例とする。

*   **責務**:
    *   特定のLiDARモデル（例: HDL-64E）の物理的制約を定義する。これには、チャンネル数（64ch）、各チャンネルの垂直角度、スキャンシーケンスなどが含まれる。
    *   点群データから各スキャンポイント（方位角、仰俯角）における距離と反射強度を読み取り、それに対応する理想的な反射波形を生成する。

*   **主要な内部パラメータ (HDL-64E)**:
    *   `fire_angles`: 64チャンネルそれぞれの垂直補正角（Vertical Correction Factor）と水平補正角（Rotational Correction Factor）。
    *   `output_channels`: 出力するチャンネル数。`64`または`32`（間引き）を選択可能。
    *   `output_horizontal_resolution_deg`: 出力解像度。例: `0.0818`度は、360度スキャンで約4400サンプルに相当する。
    *   `scan_mode`: `'vertical'`または`'horizontal'`。スキャン時のタイムスタンプ計算方法を決定する。

#### 2.3. HFR攻撃モデル (HFR Attack Models)

HFR攻撃モデルは `spaal2/core/dummy_spoofer/` 内で定義されている。`dummy_spoofer_adaptive_hfr.py` はその一例である。

*   **責務**:
    *   指定された周波数とパルス幅で、高周波の偽のレーザーパルス列を生成する。
    *   LiDARが特定の角度（`spoofer-angle`, `spoofer-altitude`）で正当なパルスを検出したことをトリガーとして、攻撃を開始する。
    *   攻撃信号は、指定された角度範囲（`spoofer-width-deg`）内でLiDAR信号に重畳される。

*   **主要な内部パラメータ**:
    *   `frequency`: HFR攻撃の周波数 (Hz)。例: `10e6` (10MHz)。
    *   `duration`: 攻撃の継続時間。
    *   `spoofer_distance_m`: LiDARと攻撃装置との間の仮想的な距離。これにより信号の遅延が決まる。
    *   `pulse_width`: 偽パルス一つ一つの幅。
    *   `amplitude`: 偽パルスの強度。

#### 2.4. 信号復元パイプライン (Signal Reconstruction Pipeline)

復元プロセスは `reconstruction/run_pipeline.py` によって統括される。

*   **責務**:
        1.  **周波数特定**: `hfr_frequency_identifier_fourier.py` を用い、攻撃を受けた信号のFFT（高速フーリエ変換）解析を行い、HFR攻撃の支配的な周波数を特定する。
        2.  **信号復元**: `peak_interval_reconstructor.py` を用い、特定された周波数に対応する周期で出現する偽ピークを検出し、信号から除去する。

*   **主要な実行時パラメータ**:
    *   `input_npz`: 攻撃を受けた`hist-matrix`ファイルへのパス。
    *   `output_npz`: 復元された`hist-matrix`を保存するパス。
    *   `--id-threshold`: 周波数特定時にFFT解析の対象とする信号を判断するためのピーク強度閾値。
    *   `--recon-tolerance`: 偽ピークを検出する際の周期の一致に関する許容誤差（ns）。
    *   `--recon-min-run`: 連続して何個の偽ピークが検出された場合に攻撃とみなすかの最小数。

#### 2.5. 評価 (Evaluation)

復元精度は `evaluation/evaluate_reconstruction.py` スクリプトで評価される。

*   **責務**:
    *   復元された`hist-matrix`から各スキャンポイントの距離を再計算する。
    *   `answer_matrix`（攻撃前の真の距離データ）と比較し、誤差を計算する。

*   **主要な実行時パラメータ**:
    *   `reconstructed_hist_matrix`: 復元された`hist-matrix`ファイルへのパス。
    *   `answer_matrix`: 正解データを含む`hist-matrix`ファイルへのパス。
    *   `--method`: 評価指標。`mae` (Mean Absolute Error) または `mse` (Mean Squared Error)。

---

### 3. 実行コマンド例 (Example Execution Commands)

以下に、uv（`uv run`）を使用した典型的な実験ワークフローのコマンド例を示す。

1.  **HFR攻撃ありの`hist-matrix`を生成**:
    ```bash
    uv run python datasets_generator/hist_matrix_generator.py \
        --lidar-type PCD_HDL64E \
        --pcd-directory /path/to/your/kitti/dataset \
        --output-dir ./pcd_datasets/attacked_data \
        --num-frames 10 \
        --spoofer-type adaptive_hfr_perturbation \
        --spoofer-angle 0.0 \
        --spoofer-altitude 2.0 \
        --spoofer-width-deg 90.0 \
        --output-horizontal-resolution-deg 0.0818
    ```

2.  **信号を復元**:
    ```bash
    uv run python reconstruction/run_pipeline.py \
        ./pcd_datasets/attacked_data/your_sample_token.npz \
        ./pcd_datasets/reconstructed_data/your_sample_token_recon.npz \
        --id-threshold 0.1 \
        --recon-tolerance 1.5
    ```

3.  **復元精度を評価**:
    *   まず、攻撃なしの正解データ (`ground_truth`) を生成する。
        ```bash
        uv run python datasets_generator/hist_matrix_generator.py \
            --lidar-type PCD_HDL64E \
            --pcd-directory /path/to/your/kitti/dataset \
            --output-dir ./pcd_datasets/ground_truth \
            --num-frames 10 \
            --spoofer-type off \
            --output-horizontal-resolution-deg 0.0818
        ```
    *   評価を実行する。
        ```bash
        uv run python evaluation/evaluate_reconstruction.py \
            ./pcd_datasets/reconstructed_data/your_sample_token_recon.npz \
            ./pcd_datasets/ground_truth/your_sample_token.npz \
            --method mae
    ```
