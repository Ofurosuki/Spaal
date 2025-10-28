#!/bin/bash

# Evaluate denoised bl2 datasets against ground truth
# This script processes all SYNC_ANGLE variations

# Base directories
GT_DIR="D:/eval_cvpr2026/data/gt_bl2"
DENOISED_BASE_DIR="D:/eval_cvpr2026/denoised_bl2"
OUTPUT_BASE_DIR="D:/eval_cvpr2026/evaluation_results"

# SYNC_ANGLE values to evaluate
SYNC_ANGLES=(0_2 0_8 1 2 5 11 22 45)

# Evaluation parameters
THRESHOLD=0.5  # meters
MIN_GT_DISTANCE=0.0  # meters

# Angle restriction parameters (matching spoofer attack cone)
EVAL_ANGLE=0  # 0 degrees = front (user-facing coordinate system)
EVAL_WIDTH=90  # 90 degrees width (matching spoofer-width-deg default)

echo "Starting evaluation of denoised datasets..."
echo "=================================================================="
echo "Evaluation will be restricted to angle: ${EVAL_ANGLE}° ± ${EVAL_WIDTH}/2° (width: ${EVAL_WIDTH}°)"
echo ""

# Loop through each SYNC_ANGLE directory
for SYNC_ANGLE in "${SYNC_ANGLES[@]}"
do
  DENOISED_DIR="${DENOISED_BASE_DIR}/sync_${SYNC_ANGLE}"
  OUTPUT_DIR="${OUTPUT_BASE_DIR}/sync_${SYNC_ANGLE}"

  # Check if denoised directory exists
  if [ ! -d "${DENOISED_DIR}" ]; then
    echo "Skipping sync_${SYNC_ANGLE}: denoised directory not found"
    continue
  fi

  echo ""
  echo "Evaluating SYNC_ANGLE=${SYNC_ANGLE}..."
  echo "GT:       ${GT_DIR}"
  echo "Denoised: ${DENOISED_DIR}"
  echo "Output:   ${OUTPUT_DIR}"

  uv run python evaluation/compare_directories_bl2.py \
    --gt-dir "${GT_DIR}" \
    --denoised-dir "${DENOISED_DIR}" \
    --threshold "${THRESHOLD}" \
    --min-gt-distance "${MIN_GT_DISTANCE}" \
    --eval-angle "${EVAL_ANGLE}" \
    --eval-width "${EVAL_WIDTH}" \
    --plot \
    --save-csv \
    --output-dir "${OUTPUT_DIR}"

  if [ $? -eq 0 ]; then
    echo "✓ Completed evaluation for SYNC_ANGLE=${SYNC_ANGLE}"
  else
    echo "✗ Failed evaluation for SYNC_ANGLE=${SYNC_ANGLE}"
  fi
done

echo ""
echo "=================================================================="
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_BASE_DIR}/"
echo ""
echo "Summary of outputs for each sync angle:"
echo "  - evaluation_results.csv: Per-frame metrics"
echo "  - overall_error_distribution.png: Error histogram"
echo "  - per_frame_metrics.png: MAE and accuracy over time"
