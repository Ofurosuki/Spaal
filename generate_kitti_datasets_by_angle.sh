#!/bin/bash
# This script generates KITTI datasets for different sync angles.

ANGLES=(2 5)

for angle in "${ANGLES[@]}"
do
  # Replace dot with underscore for directory name
  dir_angle=$(echo "$angle" | tr '.' '_')
  #output_dir="/home/yoshida/dataset_spaal/val/kitti/horizontal/${dir_angle}"

  echo "Generating dataset for sync-angle: ${angle}"
  #echo "Output directory: ${output_dir}"

  uv run python datasets_generator/hist_matrix_generator.py \
    --lidar-type PCD_HDL64E \
    --pcd-directory /data2/yoshida/100_sample_test \
    --num-frames 100 \
    --output-dir /data2/yoshida/kitti_100/attacked_64/"${dir_angle}" \
    --time-resolution-ns 1.0 \
    --scan-mode horizontal \
    --sync-angle "${angle}" \
    --start-frame 0 \
    --output-horizontal-resolution-deg 0.2 \
    --spoofer-angle 90 \
    --output-channels 64

  echo "Dataset generation for sync-angle ${angle} complete!"
  echo "----------------------------------------------------"
done

echo "All dataset generations are complete."
