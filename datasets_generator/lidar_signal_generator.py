import numpy as np
import os
from spaal2.core import (
    DummyLidarVLP16, PreciseDuration, DummyOutdoor, apply_noise, gen_sunlight,
    DummySpooferAdaptiveHFRWithPerturbation, # または DummySpooferContinuousPulseWithPerturbation
)

class LidarSignalDatasetGenerator:
    def __init__(self, lidar_model: str = "VLP16", output_dir: str = "./datasets",
                 outdoor_distance: float = 50.0, outdoor_ratio: float = 0.8,
                 spoofer_type: str = "adaptive_hfr_perturbation",
                 spoofer_frequency: float = 10 * 1e6,
                 spoofer_duration_ms: float = 20,
                 spoofer_distance_m: float = 10.0,
                 spoofer_pulse_width_ns: float = 5,
                 spoofer_perturbation_ns: float = 20.0,
                 spoofer_amplitude_range: tuple[float, float] = (5.0, 9.0), # (min, max)
                 lidar_amplitude_range: tuple[float, float] = (1.0, 7.0), # (min, max)
                 lidar_pulse_width_ns: float = 5,
                 noise_ratio: float = 0.1,
                 sunlight_mean: float = 0.5):
        if lidar_model == "VLP16":
            self.lidar = DummyLidarVLP16(amplitude=lidar_amplitude_range[0], pulse_width=PreciseDuration(nanoseconds=lidar_pulse_width_ns))
            self.channels = 16
            self.horizontal_resolution = 1800  # 360 / 0.2
            self.samples_per_scan = self.lidar.accept_window.in_nanoseconds
        else:
            raise ValueError(f"Unknown LiDAR model: {lidar_model}")

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.outdoor = DummyOutdoor(outdoor_distance, outdoor_ratio)
        self.noise_ratio = noise_ratio
        self.sunlight_mean = sunlight_mean

        self.lidar_amplitude_range = lidar_amplitude_range
        self.spoofer_amplitude_range = spoofer_amplitude_range

        if spoofer_type == "adaptive_hfr_perturbation":
            self.spoofer = DummySpooferAdaptiveHFRWithPerturbation(
                frequency=spoofer_frequency,
                duration=PreciseDuration(milliseconds=spoofer_duration_ms),
                spoofer_distance_m=spoofer_distance_m,
                pulse_width=PreciseDuration(nanoseconds=spoofer_pulse_width_ns),
                perturbation_ns=spoofer_perturbation_ns,
                amplitude=spoofer_amplitude_range[0], # 初期値として範囲の最小値を使用
            )
        elif spoofer_type == "continuous_pulse_perturbation":
            from spaal2.core import DummySpooferContinuousPulseWithPerturbation
            self.spoofer = DummySpooferContinuousPulseWithPerturbation(
                frequency=spoofer_frequency,
                pulse_width=PreciseDuration(nanoseconds=spoofer_pulse_width_ns),
                perturbation_ns=spoofer_perturbation_ns,
                amplitude=spoofer_amplitude_range[0], # 初期値として範囲の最小値を使用
            )
        else:
            raise ValueError(f"Unknown spoofer type: {spoofer_type}")

    def generate(self, num_frames: int, filename_prefix: str = "lidar_signal"):
        """
        指定されたフレーム数のLiDAR信号データを生成し、.npyファイルとして保存します。

        Parameters
        ----------
        num_frames : int
            生成するフレーム数。
        filename_prefix : str, optional
            保存するファイル名のプレフィックス, by default "lidar_signal"
        """
        all_frames_data = np.zeros((num_frames, self.channels, self.horizontal_resolution, self.samples_per_scan), dtype=np.float32)

        for frame_num in range(num_frames):
            print(f"Generating frame {frame_num + 1}/{num_frames}...")
            # フレームごとにLiDARをリセット
            current_lidar = self.lidar.new_frame(base_timestamp=PreciseDuration(nanoseconds=frame_num * 10**9))
            is_triggered = False

            # データを格納する配列を初期化
            frame_data = np.zeros((self.channels, self.horizontal_resolution, self.samples_per_scan), dtype=np.float32)

            try:
                while True:
                    config, signal = current_lidar.scan()

                    # 各スキャンでLiDARとスプーファーの振幅をランダムに設定
                    lidar_amp = np.random.uniform(self.lidar_amplitude_range[0], self.lidar_amplitude_range[1])
                    spoofer_amp = np.random.uniform(self.spoofer_amplitude_range[0], self.spoofer_amplitude_range[1])
                    current_lidar.set_amplitude(lidar_amp)
                    self.spoofer.set_amplitude(spoofer_amp)

                    # HFR攻撃のトリガー条件 (例: ahfr_with_perturbation.pyから)
                    if config.altitude == 900 and abs(config.azimuth - 1000) < 20:
                        self.spoofer.trigger(config, signal) # spooferをトリガー
                        is_triggered = True

                    # 正規信号に環境ノイズを適用
                    signal = apply_noise(self.outdoor.apply(signal), ratio=self.noise_ratio)
                    
                    # 太陽光ノイズを追加
                    sunlight_noise = gen_sunlight(len(signal), self.sunlight_mean)
                    signal = signal + sunlight_noise
                    
                    # スプーファー信号を取得
                    external_signal = apply_noise(self.spoofer.get_range_signal(config.start_timestamp, config.accept_duration), ratio=0.01)
                    
                    # 信号の合成
                    signal = np.choose(
                        signal < external_signal,
                        [signal, external_signal]
                    )
                    signal = np.clip(signal, 0, 9) # 信号値を0-9の範囲にクリップ

                    horizontal_index = config.azimuth // 20  # 0.2度ステップなので20を掛ける
                    vertical_index = current_lidar.vertical_angles.index(config.altitude / 100)

                    if horizontal_index < self.horizontal_resolution:
                        frame_data[vertical_index, horizontal_index, :] = signal

            except StopIteration:
                pass  # 1フレームの終わり

            all_frames_data[frame_num, :, :, :] = frame_data

        # ファイルに保存
        output_filename = os.path.join(self.output_dir, f"{filename_prefix}.npy")
        np.save(output_filename, all_frames_data)
        print(f"Saved all frames to {output_filename}")

if __name__ == '__main__':
    # 使用例
    generator = LidarSignalDatasetGenerator(lidar_model="VLP16", output_dir="./lidar_datasets")
    generator.generate(num_frames=5)  # 5フレーム分のデータを生成
