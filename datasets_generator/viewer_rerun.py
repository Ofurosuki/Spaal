import rerun as rr
import numpy as np
import glob
import os

rr.init("kitti_visualizer", spawn=True)

# .bin ファイルの検索パターンを定義
search_pattern = "D:/cvpr2026_data/test/4.0/*/attacked/bin/*.bin"
bin_files = sorted(glob.glob(search_pattern))

if not bin_files:
    print(f"No .bin files found matching the pattern: {search_pattern}")
else:
    print(f"Found {len(bin_files)} files to visualize.")

# データを読み込んで可視化
for i, file_path in enumerate(bin_files):
    rr.set_time_sequence("frame_idx", i)
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
    rr.log("lidar", rr.Points3D(points[:, :3], colors=[255, 255, 255]))
    print(f"Displayed frame {i}: {os.path.basename(file_path)}")
