#!/usr/bin/env python3
"""
CenterPoint 用に .bin ファイルを前処理する。

KITTI .bin (Nx4: x, y, z, intensity[0-1]) を
CenterPoint/nuScenes 形式 (Nx5: x, y, z, intensity[0-255], ring) へ変換する。

変換内容:
  - intensity (列3) を 255 倍
  - ring 列 (列4) を 0 で追加
"""

import numpy as np
import argparse
import os
import glob
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


def convert_bin(src_path: str, dst_path: str) -> None:
    pts = np.fromfile(src_path, dtype=np.float32).reshape(-1, 4)

    # intensity を 255 倍
    pts[:, 3] = pts[:, 3] * 255.0

    # ring 列 (0 埋め) を追加 → Nx5
    ring = np.zeros((pts.shape[0], 1), dtype=np.float32)
    pts5 = np.concatenate([pts, ring], axis=1)  # (N, 5)

    pts5.tofile(dst_path)


def main():
    parser = argparse.ArgumentParser(
        description="KITTI .bin (Nx4) を CenterPoint 用 (Nx5) に変換する"
    )
    parser.add_argument("input_dir",  help="入力 .bin ディレクトリ")
    parser.add_argument("output_dir", help="出力 .bin ディレクトリ")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(input_dir.glob("*.bin"))
    if not bin_files:
        print(f"[WARN] .bin ファイルが見つかりません: {input_dir}")
        return

    print(f"[INFO] {len(bin_files)} ファイルを変換: {input_dir} -> {output_dir}")
    for src in tqdm(bin_files):
        dst = output_dir / src.name
        convert_bin(str(src), str(dst))

    print(f"[INFO] 完了: {output_dir}")


if __name__ == "__main__":
    main()
