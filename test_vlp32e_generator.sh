#!/bin/bash
# Test script for VLP-32c dataset generation

# Example command to generate VLP-32c dataset with vertical scan mode
uv run python datasets_generator/hist_matrix_generator.py \
  --lidar-type PCD_VLP32c \
  --pcd-directory ./nuscenes_data \
  --num-frames 1 \
  --output-dir ./lidar_datasets_vlp32c \
  --scan-mode vertical \
  --sync-angle 2 4 \
  --spoofer-angle 0 \
  --spoofer-altitude 50

echo "VLP-32c dataset generation complete!"