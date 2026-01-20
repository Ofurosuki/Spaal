#!/bin/bash

# Template Subtraction Denoising for evaluation data
# This script processes all SYNC_ANGLE variations

# Base directories
<<<<<<< HEAD
INPUT_BASE_DIR="/data2/yoshida/kitti_100/kitti/horizontal"
OUTPUT_BASE_DIR="/data2/yoshida/kitti_100/baseline_denoised_64"

# SYNC_ANGLE values to process
#SYNC_ANGLES=(0_2 0_8 1 2 5 11 22 45)
SYNC_ANGLES=(0_2 1)
=======
INPUT_BASE_DIR="D:/eval_cvpr2026/data/attacked_bl2"
OUTPUT_BASE_DIR="D:/baseline/bl2"

# SYNC_ANGLE values to process
SYNC_ANGLES=(0_8)
>>>>>>> 1bbb0ae6e35556cfcca48d13dbacc006f6112dd7

# Denoising parameters
MIN_TEMPLATE_SAMPLES=2
MIN_PEAK_THRESHOLD=0.01
CHUNK_SIZE=20

echo "Starting Template Subtraction Denoising..."
echo "=================================================================="
echo "Minimum template samples: ${MIN_TEMPLATE_SAMPLES}"
echo "Minimum peak threshold: ${MIN_PEAK_THRESHOLD}"
echo "Chunk size: ${CHUNK_SIZE} samples"
echo ""

# Loop through each SYNC_ANGLE
for SYNC_ANGLE in "${SYNC_ANGLES[@]}"
do
<<<<<<< HEAD
  INPUT_DIR="${INPUT_BASE_DIR}/${SYNC_ANGLE}"
  OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SYNC_ANGLE}"
=======
  INPUT_DIR="${INPUT_BASE_DIR}/sync_${SYNC_ANGLE}"
  OUTPUT_DIR="${OUTPUT_BASE_DIR}/sync_${SYNC_ANGLE}"
>>>>>>> 1bbb0ae6e35556cfcca48d13dbacc006f6112dd7

  # Check if input directory exists
  if [ ! -d "${INPUT_DIR}" ]; then
    echo "Skipping sync_${SYNC_ANGLE}: directory not found"
    continue
  fi

  echo ""
  echo "Processing SYNC_ANGLE=${SYNC_ANGLE}..."
  echo "Input:  ${INPUT_DIR}"
  echo "Output: ${OUTPUT_DIR}"

  uv run python baseline_denoising/template_subtraction_denoiser.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --min-template-samples "${MIN_TEMPLATE_SAMPLES}" \
    --min-peak-threshold "${MIN_PEAK_THRESHOLD}" \
    --chunk-size "${CHUNK_SIZE}"

  if [ $? -eq 0 ]; then
    echo "✓ Completed SYNC_ANGLE=${SYNC_ANGLE}"
  else
    echo "✗ Failed for SYNC_ANGLE=${SYNC_ANGLE}"
  fi
done

echo ""
echo "=================================================================="
echo "Template Subtraction Denoising complete!"
echo "Denoised data saved to: ${OUTPUT_BASE_DIR}/"
