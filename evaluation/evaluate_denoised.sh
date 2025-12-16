#!/bin/bash

# Evaluate denoised results against ground truth
# This script evaluates all SYNC_ANGLE variations
# Usage: ./evaluate_denoised.sh [--single-sample]

# Parse command line arguments
SINGLE_SAMPLE=false
if [[ "$1" == "--single-sample" ]]; then
  SINGLE_SAMPLE=true
fi

# Base directories
GT_DIR="D:/eval_cvpr2026/data/gt_bl2"
#DENOISED_BASE_DIR="D:/eval_cvpr2026/data/attacked_bl2"
#DENOISED_BASE_DIR="D:/regression"
DENOISED_BASE_DIR="D:/1108_ablation/dim24/bl2"
#DENOISED_BASE_DIR="D:/kitti_denoised_bl2"
#RESULTS_DIR="D:/eval_cvpr2026/evaluation_results/attacked"
RESULTS_DIR="D:/1108_ablation/dim24"

# Evaluation parameters
THRESHOLD=0.5  # meters
MIN_GT_DISTANCE=0.0  # meters

# Angle restriction parameters (matching spoofer attack cone)
EVAL_ANGLE=0  # 0 degrees = front (user-facing coordinate system)
EVAL_WIDTH=90  # 90 degrees width (matching spoofer-width-deg default)

# SYNC_ANGLE values to evaluate
if [ "$SINGLE_SAMPLE" = true ]; then
  SYNC_ANGLES=(1)  # Only first SYNC_ANGLE for debugging
  MAX_SAMPLES_ARG="--max-samples 1"
  echo "DEBUG MODE: Evaluating only 1 sample from SYNC_ANGLE=1"
else
  #SYNC_ANGLES=(0_2 0_8 1 2 5 11 22 45)
  SYNC_ANGLES=(1)
  #SYNC_ANGLES=(0_8)
  MAX_SAMPLES_ARG=""
fi

echo "Starting evaluation of denoised results..."
echo "=================================================================="
echo "Ground Truth Directory: ${GT_DIR}"
echo "Threshold: ${THRESHOLD}m"
echo "Min GT Distance: ${MIN_GT_DISTANCE}m"
echo "Evaluation Angle: ${EVAL_ANGLE}° ± ${EVAL_WIDTH}/2° (width: ${EVAL_WIDTH}°)"
echo "=================================================================="

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Summary file
SUMMARY_FILE="${RESULTS_DIR}/summary.txt"
echo "Evaluation Summary" > "${SUMMARY_FILE}"
echo "==================" >> "${SUMMARY_FILE}"
echo "Date: $(date)" >> "${SUMMARY_FILE}"
echo "Threshold: ${THRESHOLD}m" >> "${SUMMARY_FILE}"
echo "Min GT Distance: ${MIN_GT_DISTANCE}m" >> "${SUMMARY_FILE}"
echo "Evaluation Angle: ${EVAL_ANGLE}° ± ${EVAL_WIDTH}/2° (width: ${EVAL_WIDTH}°)" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Loop through each SYNC_ANGLE
for SYNC_ANGLE in "${SYNC_ANGLES[@]}"
do
  DENOISED_DIR="${DENOISED_BASE_DIR}/sync_${SYNC_ANGLE}"
  OUTPUT_DIR="${RESULTS_DIR}/sync_${SYNC_ANGLE}"

  # Check if denoised directory exists
  if [ ! -d "${DENOISED_DIR}" ]; then
    echo ""
    echo "Skipping sync_${SYNC_ANGLE}: denoised directory not found"
    echo "sync_${SYNC_ANGLE}: NOT FOUND" >> "${SUMMARY_FILE}"
    continue
  fi

  echo ""
  echo "Evaluating SYNC_ANGLE=${SYNC_ANGLE}..."
  echo "Denoised: ${DENOISED_DIR}"
  echo "Output:   ${OUTPUT_DIR}"

  mkdir -p "${OUTPUT_DIR}"

  # Run evaluation
  uv run python evaluation/compare_directories_bl2.py \
    --gt-dir "${GT_DIR}" \
    --denoised-dir "${DENOISED_DIR}" \
    --threshold "${THRESHOLD}" \
    --min-gt-distance "${MIN_GT_DISTANCE}" \
    --eval-angle "${EVAL_ANGLE}" \
    --eval-width "${EVAL_WIDTH}" \
    --plot \
    --save-csv \
    --output-dir "${OUTPUT_DIR}" \
    ${MAX_SAMPLES_ARG} \
    > "${OUTPUT_DIR}/evaluation_log.txt" 2>&1

  if [ $? -eq 0 ]; then
    echo "✓ Completed SYNC_ANGLE=${SYNC_ANGLE}"

    # Extract results from log and add to summary
    MAE=$(grep "Overall Mean Absolute Error:" "${OUTPUT_DIR}/evaluation_log.txt" | awk '{print $5}')
    ACC=$(grep "Overall Accuracy" "${OUTPUT_DIR}/evaluation_log.txt" | awk '{print $6}' | sed 's/%//')

    echo "sync_${SYNC_ANGLE}: MAE=${MAE}m, Accuracy=${ACC}%" >> "${SUMMARY_FILE}"
    echo "  MAE: ${MAE}m, Accuracy: ${ACC}%"
  else
    echo "✗ Failed for SYNC_ANGLE=${SYNC_ANGLE}"
    echo "sync_${SYNC_ANGLE}: FAILED" >> "${SUMMARY_FILE}"
  fi
done

echo ""
echo "=================================================================="
echo "Evaluation complete!"
echo "Results saved in: ${RESULTS_DIR}/"
echo "Summary: ${SUMMARY_FILE}"
echo "=================================================================="

# Display summary
echo ""
echo "=== EVALUATION SUMMARY ==="
cat "${SUMMARY_FILE}"
