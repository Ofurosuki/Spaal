#!/bin/bash

# Template Subtraction Denoising Full Pipeline
# This script: denoise -> convert to bin -> inference -> evaluate

set -e  # Exit on error
########################################################################################
# Base directories
#INPUT_BASE_DIR="/data2/yoshida/kitti_100/attacked_64"
INPUT_BASE_DIR="/data2/yoshida/nus_100/attacked_bl2"
BASELINE_DENOISED_DIR="/data2/yoshida/nus_100/baseline_denoised_64"
BASELINE_BIN_DIR="/data2/yoshida/nus_100/baseline_bin_64"
FORMAT="kitti"


# SYNC_ANGLE values to process
#SYNC_ANGLES=(22 5 2 1 0_8 45)
SYNC_ANGLES=(11)
# Full list: (0_2 0_8 1 2 5 11 22 45)

# Denoising parameters
MIN_TEMPLATE_SAMPLES=2
MIN_PEAK_THRESHOLD=0.05
CHUNK_SIZE=20

########################################################################################

# Docker configuration
DOCKER_CONTAINER_NAME="sharp_hodgkin"
MODEL_CONFIG="configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
MODEL_CHECKPOINT="./nuscenes/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"
GT_LABEL_DIR="/data2/yoshida/label_kitti/training/label_2"

# Timestamp for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/data2/yoshida/kitti_100/baseline_evaluation_results"
mkdir -p "${RESULTS_DIR}"
SUMMARY_FILE="${RESULTS_DIR}/baseline_template_summary_${TIMESTAMP}.txt"

echo "==========================================================" | tee "${SUMMARY_FILE}"
echo "Template Subtraction Denoising - Full Evaluation Pipeline" | tee -a "${SUMMARY_FILE}"
echo "==========================================================" | tee -a "${SUMMARY_FILE}"
echo "Timestamp: ${TIMESTAMP}" | tee -a "${SUMMARY_FILE}"
echo "Input:     ${INPUT_BASE_DIR}" | tee -a "${SUMMARY_FILE}"
echo "Output:    ${BASELINE_DENOISED_DIR}" | tee -a "${SUMMARY_FILE}"
echo "Angles:    ${SYNC_ANGLES[@]}" | tee -a "${SUMMARY_FILE}"
echo "==========================================================" | tee -a "${SUMMARY_FILE}"
echo "" | tee -a "${SUMMARY_FILE}"

# Loop through each SYNC_ANGLE
for SYNC_ANGLE in "${SYNC_ANGLES[@]}"
do
  # Docker path (relative to Docker's working directory)
  DOCKER_BIN_DIR="./nuscenes/nus_100/baseline_bin_64/sync_${SYNC_ANGLE}"
  DOCKER_RESULTS_JSON="${DOCKER_BIN_DIR}/predictions_baseline_${SYNC_ANGLE}_${TIMESTAMP}.json"
  echo ""
  echo "==========================================================" | tee -a "${SUMMARY_FILE}"
  echo "Processing SYNC_ANGLE: ${SYNC_ANGLE}" | tee -a "${SUMMARY_FILE}"
  echo "==========================================================" | tee -a "${SUMMARY_FILE}"

  INPUT_DIR="${INPUT_BASE_DIR}/sync_${SYNC_ANGLE}"
  DENOISED_DIR="${BASELINE_DENOISED_DIR}/${SYNC_ANGLE}"
  BIN_DIR="${BASELINE_BIN_DIR}/sync_${SYNC_ANGLE}"

  # Check if input directory exists
  if [ ! -d "${INPUT_DIR}" ]; then
    echo "⚠ Skipping ${SYNC_ANGLE}: Input directory not found" | tee -a "${SUMMARY_FILE}"
    continue
  fi

  # --- STEP 1: Template Subtraction Denoising ---
  # echo ""
  # echo "--- Step 1: Template Subtraction Denoising ---" | tee -a "${SUMMARY_FILE}"
  # echo "Input:  ${INPUT_DIR}" | tee -a "${SUMMARY_FILE}"
  # echo "Output: ${DENOISED_DIR}" | tee -a "${SUMMARY_FILE}"

  # uv run python baseline_denoising/template_subtraction_denoiser.py \
  #   --input-dir "${INPUT_DIR}" \
  #   --output-dir "${DENOISED_DIR}" \
  #   --min-template-samples "${MIN_TEMPLATE_SAMPLES}" \
  #   --min-peak-threshold "${MIN_PEAK_THRESHOLD}" \
  #   --chunk-size "${CHUNK_SIZE}"

  # if [ $? -ne 0 ]; then
  #   echo "✗ Denoising failed for ${SYNC_ANGLE}" | tee -a "${SUMMARY_FILE}"
  #   continue
  # fi
  # echo "✓ Denoising completed" | tee -a "${SUMMARY_FILE}"

  # # --- STEP 2: Convert bl2 to bin ---
  # echo ""
  # echo "--- Step 2: Converting bl2 to bin ---" | tee -a "${SUMMARY_FILE}"
  # echo "Input:  ${DENOISED_DIR}" | tee -a "${SUMMARY_FILE}"
  # echo "Output: ${BIN_DIR}" | tee -a "${SUMMARY_FILE}"

  # uv run python datasets_generator/bl2_to_bin_converter.py \
  #   --input-dir "${DENOISED_DIR}" \
  #   --output-dir "${BIN_DIR}" \
  #   --amplitude-to-intensity-ratio 0.08 \
  #   --format "${FORMAT}" 

  # if [ $? -ne 0 ]; then
  #   echo "✗ Conversion failed for ${SYNC_ANGLE}" | tee -a "${SUMMARY_FILE}"
  #   continue
  # fi
  # echo "✓ Conversion completed" | tee -a "${SUMMARY_FILE}"

  # # Check generated files
  # NUM_BIN_FILES=$(ls -1 "${BIN_DIR}"/*.bin 2>/dev/null | wc -l)
  # echo "Generated ${NUM_BIN_FILES} .bin files" | tee -a "${SUMMARY_FILE}"

  # # --- STEP 3: Inference ---
  # echo ""
  # echo "--- Step 3: Running inference ---" | tee -a "${SUMMARY_FILE}"

  

  # docker exec ${DOCKER_CONTAINER_NAME} python demo/my_inference.py \
  #   "${DOCKER_BIN_DIR}" \
  #   "${MODEL_CONFIG}" \
  #   "${MODEL_CHECKPOINT}" \
  #   "--output-json" "${DOCKER_RESULTS_JSON}"

  # if [ $? -ne 0 ]; then
  #   echo "✗ Inference failed for ${SYNC_ANGLE}" | tee -a "${SUMMARY_FILE}"
  #   continue
  # fi
  # echo "✓ Inference completed" | tee -a "${SUMMARY_FILE}"

  # --- STEP 4: Evaluation ---
  echo ""
  echo "--- Step 4: Evaluating results ---" | tee -a "${SUMMARY_FILE}"

  # Convert Docker path to host path
  HOST_RESULTS_JSON="${BASELINE_BIN_DIR}/sync_${SYNC_ANGLE}/predictions_baseline_${SYNC_ANGLE}_${TIMESTAMP}.json"
  echo "Results JSON: ${HOST_RESULTS_JSON}" | tee -a "${SUMMARY_FILE}"

  if [ ! -f "${HOST_RESULTS_JSON}" ]; then
    echo "✗ Results JSON not found: ${HOST_RESULTS_JSON}" | tee -a "${SUMMARY_FILE}"
    continue
  fi

  echo "" | tee -a "${SUMMARY_FILE}"
  echo "--- Evaluation for SYNC_ANGLE=${SYNC_ANGLE} ---" | tee -a "${SUMMARY_FILE}"

  uv run python kitti_eval_package/calculate_map_nuscenes_style.py \
    "${HOST_RESULTS_JSON}" \
    --gt-label-dir "${GT_LABEL_DIR}" \
    --filter-kitti-range | tee -a "${SUMMARY_FILE}"

  if [ $? -eq 0 ]; then
    echo "✓ Evaluation completed for ${SYNC_ANGLE}" | tee -a "${SUMMARY_FILE}"
  else
    echo "✗ Evaluation failed for ${SYNC_ANGLE}" | tee -a "${SUMMARY_FILE}"
  fi

  echo "" | tee -a "${SUMMARY_FILE}"
  echo "==========================================================" | tee -a "${SUMMARY_FILE}"

done

echo ""
echo "==========================================================" | tee -a "${SUMMARY_FILE}"
echo "Pipeline complete!" | tee -a "${SUMMARY_FILE}"
echo "Summary saved to: ${SUMMARY_FILE}" | tee -a "${SUMMARY_FILE}"
echo "==========================================================" | tee -a "${SUMMARY_FILE}"
