#!/bin/bash

# Array of SYNC_ANGLE values to generate datasets for
SYNC_ANGLES=(0.2 0.8 1 2 5 11 22 45)

# Base output directory
BASE_OUTPUT_DIR="D:/eval_cvpr2026/data/attacked_bl2"

# Input directory
INPUT_DIR="D:/eval_cvpr2026/data/point_clouds_gt"

echo "Starting VLP-32c dataset generation for multiple SYNC_ANGLE values..."
echo "=================================================================="

# Loop through each SYNC_ANGLE value
for SYNC_ANGLE in "${SYNC_ANGLES[@]}"
do
  # Create output directory name based on SYNC_ANGLE (replace . with _)
  SYNC_ANGLE_CLEAN=$(echo "${SYNC_ANGLE}" | sed 's/\./_/g')
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/sync_${SYNC_ANGLE_CLEAN}"

  echo ""
  echo "Generating dataset with SYNC_ANGLE=${SYNC_ANGLE}..."
  echo "Output directory: ${OUTPUT_DIR}"

  uv run python datasets_generator/hist_matrix_generator.py \
    --lidar-type PCD_VLP32c \
    --pcd-directory "${INPUT_DIR}" \
    --num-frames 81 \
    --output-dir "${OUTPUT_DIR}" \
    --scan-mode horizontal \
    --sync-angle "${SYNC_ANGLE}" \
    --spoofer-angle 0 \
    --spoofer-altitude 50

  if [ $? -eq 0 ]; then
    echo "✓ Completed SYNC_ANGLE=${SYNC_ANGLE}"
  else
    echo "✗ Failed for SYNC_ANGLE=${SYNC_ANGLE}"
  fi
done

echo ""
echo "=================================================================="
echo "All VLP-32c dataset generation complete!"
echo "Generated ${#SYNC_ANGLES[@]} datasets in ${BASE_OUTPUT_DIR}/"