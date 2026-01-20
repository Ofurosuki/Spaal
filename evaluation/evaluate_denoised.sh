#!/bin/bash

# Evaluate denoised results against ground truth
# This script evaluates all SYNC_ANGLE variations

# Base directories
GT_DIR="/data2/yoshida/kitti_100/kitti_no_spoofer_64"
#GT_DIR="/data2/yoshida/1121_data_gt"
DENOISED_BASE_DIR="/data2/yoshida/kitti_100/baseline_denoised_64"

RESULTS_DIR="/data2/yoshida/kitti_100/baseline_denoised_64/evaluation_results"

# Evaluation parameters
THRESHOLD=0.5  # meters
MIN_GT_DISTANCE=0.0  # meters

# Angle restriction parameters (matching spoofer attack cone)
EVAL_ANGLE=90  # 0 degrees = front (user-facing coordinate system) # kitti coordinate: 90 degrees nuscenes coordinate: 0 degrees
EVAL_WIDTH=90  # 90 degrees width (matching spoofer-width-deg default)

# SYNC_ANGLE values to evaluate
SYNC_ANGLES=(11 22 5 2 1 0_8 45)

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
  #DENOISED_DIR="${DENOISED_BASE_DIR}/sync_${SYNC_ANGLE}"
  #OUTPUT_DIR="${RESULTS_DIR}/sync_${SYNC_ANGLE}"
  DENOISED_DIR="${DENOISED_BASE_DIR}/${SYNC_ANGLE}"
  OUTPUT_DIR="${RESULTS_DIR}/${SYNC_ANGLE}"

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
