#!/bin/bash
# Test script for VLP-32c dataset generation

# Example command to generate VLP-32c dataset with vertical scan mode
uv run python datasets_generator/hist_matrix_generator.py \
  --lidar-type PCD_VLP32c \
  --pcd-directory ./nuscenes_data \
  --num-frames 10 \
  --output-dir D:/cvpr2026_data/test \
  --scan-mode horizontal \
  --sync-angle 4 \
  --spoofer-angle 0 \
  --spoofer-altitude 50 

echo "VLP-32c dataset generation complete!"