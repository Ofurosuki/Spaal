#!/bin/bash
# Test script for HDL-64E dataset generation with KITTI data

# Example command to generate HDL-64E dataset from KITTI bin files
uv run python datasets_generator/hist_matrix_generator.py \
  --lidar-type PCD_HDL64E \
  --pcd-directory D:/testing/velodyne \
  --num-frames 2 \
  --output-dir ./lidar_datasets_hdl64e \
  --time-resolution-ns 1.0 \
  --sync-angle 1.0 \
  --start-frame 0 \
  --output-horizontal-resolution-deg 0.2

echo "HDL-64E dataset generation complete!"
