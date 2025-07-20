import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

class LidarSignalVisualizer:
    def __init__(self, npy_file_path):
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"The specified file was not found: {npy_file_path}")
        
        self.data = np.load(npy_file_path)
        
        if self.data.ndim != 4:
            raise ValueError(f"Expected a 4D array, but got an array with shape {self.data.shape}")

        self.num_frames, self.num_channels, self.num_horizontal, self.num_samples = self.data.shape

        # 初期表示のインデックス
        self.current_frame = 0
        self.current_channel = 0
        self.current_horizontal = 0

        # プロットのセットアップ
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        plt.subplots_adjust(bottom=0.35)

        # 初期の1Dプロット
        initial_signal = self.data[self.current_frame, self.current_channel, self.current_horizontal, :]
        self.line, = self.ax.plot(initial_signal)
        self.ax.set_title(f'Frame: {self.current_frame}, Channel: {self.current_channel}, Horizontal Idx: {self.current_horizontal}')
        self.ax.set_xlabel('Sample Index (Time/Distance)')
        self.ax.set_ylabel('Signal Amplitude')
        self.ax.set_ylim(0, np.max(self.data) * 1.1 if np.max(self.data) > 0 else 10) # Y軸の範囲を適切に設定
        self.ax.grid(True)

        # フレーム選択スライダー
        ax_frame = plt.axes([0.25, 0.2, 0.65, 0.03])
        self.frame_slider = Slider(
            ax=ax_frame,
            label='Frame',
            valmin=0,
            valmax=self.num_frames - 1,
            valinit=self.current_frame,
            valstep=1
        )

        # チャンネル選択スライダー
        ax_channel = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.channel_slider = Slider(
            ax=ax_channel,
            label='Channel (Y)',
            valmin=0,
            valmax=self.num_channels - 1,
            valinit=self.current_channel,
            valstep=1
        )

        # 水平インデックス選択スライダー
        ax_horizontal = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.horizontal_slider = Slider(
            ax=ax_horizontal,
            label='Horizontal Idx (X)',
            valmin=0,
            valmax=self.num_horizontal - 1,
            valinit=self.current_horizontal,
            valstep=1
        )

        # スライダーの更新イベントに関数を接続
        self.frame_slider.on_changed(self.update)
        self.channel_slider.on_changed(self.update)
        self.horizontal_slider.on_changed(self.update)

    def update(self, val):
        self.current_frame = int(self.frame_slider.val)
        self.current_channel = int(self.channel_slider.val)
        self.current_horizontal = int(self.horizontal_slider.val)
        
        # プロットデータとタイトルを更新
        new_signal = self.data[self.current_frame, self.current_channel, self.current_horizontal, :]
        self.line.set_ydata(new_signal)
        self.ax.set_title(f'Frame: {self.current_frame}, Channel: {self.current_channel}, Horizontal Idx: {self.current_horizontal}')
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

if __name__ == '__main__':
    # 使用例
    dataset_file = './lidar_datasets/lidar_signal.npy'
    
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file not found at '{dataset_file}'")
        print("Please run 'lidar_signal_generator.py' first to generate the dataset.")
    else:
        visualizer = LidarSignalVisualizer(dataset_file)
        visualizer.show()