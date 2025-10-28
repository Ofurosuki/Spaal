#!/bin/bash

# Convert bl2 datasets to .bin files for evaluation
# This script processes all SYNC_ANGLE variations

# Base directories
BASE_INPUT_DIR="D:/eval_cvpr2026/data/attacked_bl2"
BASE_OUTPUT_DIR="D:/eval_cvpr2026/data/attacked_bin"

# SYNC_ANGLE values to process
SYNC_ANGLES=(0_2 0_8 1 2 5 11 22 45)

echo "Starting bl2 to bin conversion for evaluation data..."
echo "=================================================================="

# Loop through each SYNC_ANGLE directory
for SYNC_ANGLE in "${SYNC_ANGLES[@]}"
do
  INPUT_DIR="${BASE_INPUT_DIR}/sync_${SYNC_ANGLE}"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/sync_${SYNC_ANGLE}"

  # Check if input directory exists
  if [ ! -d "${INPUT_DIR}" ]; then
    echo "Skipping sync_${SYNC_ANGLE}: directory not found"
    continue
  fi

  echo ""
  echo "Processing SYNC_ANGLE=${SYNC_ANGLE}..."
  echo "Input:  ${INPUT_DIR}"
  echo "Output: ${OUTPUT_DIR}"

  uv run python datasets_generator/bl2_to_bin_converter.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --amplitude-to-intensity-ratio 25.5 \
    --format nuscenes

  if [ $? -eq 0 ]; then
    echo "✓ Completed SYNC_ANGLE=${SYNC_ANGLE}"
  else
    echo "✗ Failed for SYNC_ANGLE=${SYNC_ANGLE}"
  fi
done

echo ""
echo "=================================================================="
echo "Conversion complete!"
echo "Output directory: ${BASE_OUTPUT_DIR}/"
