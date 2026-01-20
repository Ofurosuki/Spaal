#!/bin/bash
# Test script for HDL-64E dataset generation with KITTI data

# Example command to generate HDL-64E dataset from KITTI bin files with vertical scan mode
# uv run python datasets_generator/hist_matrix_generator.py \
#   --lidar-type PCD_HDL64E \
#   --pcd-directory /data2/yoshida/100_sample_test \
#   --num-frames 100 \
#   --output-dir /data2/yoshida/kitti_100/attacked_64/1 \
#   --time-resolution-ns 1.0 \
#   --scan-mode horizontal \
#   --sync-angle 1.0 \
#   --start-frame 0 \
#   --output-horizontal-resolution-deg 0.2 \
#   --spoofer-angle 90 \
#   --output-channels 64

# echo "HDL-64E dataset generation complete!"

uv run python datasets_generator/hist_matrix_generator.py \
  --lidar-type PCD_HDL64E \
  --pcd-directory /data2/yoshida/100_sample_test \
  --num-frames 100 \
  --output-dir /data2/yoshida/kitti_100/kitti_no_spoofer_64 \
  --time-resolution-ns 1.0 \
  --scan-mode horizontal \
  --sync-angle 1.0 \
  --start-frame 0 \
  --output-horizontal-resolution-deg 0.2 \
  --spoofer-angle 90 \
  --output-channels 64 \
  --spoofer-type "off"

echo "HDL-64E dataset generation complete!"
