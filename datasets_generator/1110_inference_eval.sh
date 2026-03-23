#!/bin/bash

set -e  # Exit on error

#ANGLES=(0_2 0_8 1 2 5 11 22 45)
ANGLES=(5)
BASE_DIR="/data2/yoshida/kitti_100"

# Docker configuration
DOCKER_CONTAINER_NAME="sharp_hodgkin"
MODEL_CONFIG="configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
MODEL_CHECKPOINT="./nuscenes/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"

# Input bin directory (modify this to change which data to evaluate)
INPUT_BIN_BASE="${BASE_DIR}/baseline_denoised_bin"

echo "=================================================="
echo "KITTI Inference and Evaluation Pipeline"
echo "Processing ${#ANGLES[@]} angle configurations"
echo "Input bin directory: ${INPUT_BIN_BASE}"
echo "=================================================="

# Loop through each angle configuration
for ANGLE in "${ANGLES[@]}"; do
    echo ""
    echo "=================================================="
    echo "Processing ANGLE: ${ANGLE}"
    echo "=================================================="

    INPUT_BIN_DIR="${INPUT_BIN_BASE}/${ANGLE}"

    # Check if input directory exists
    if [ ! -d "${INPUT_BIN_DIR}" ]; then
        echo "Warning: Input directory does not exist: ${INPUT_BIN_DIR}"
        echo "Skipping ANGLE: ${ANGLE}"
        continue
    fi

    # --- Step 5: Inference ---
    echo ""
    echo "--- Step 5: Running inference ---"
    DOCKER_BASE_DIR="./nuscenes/kitti_100/baseline_denoised_bin/${ANGLE}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    DOCKER_RESULTS_JSON="${DOCKER_BASE_DIR}/predictions_${ANGLE}_${TIMESTAMP}.json"

    echo "Docker input path: ${DOCKER_BASE_DIR}"
    echo "Output JSON: ${DOCKER_RESULTS_JSON}"

    docker exec ${DOCKER_CONTAINER_NAME} python demo/my_inference.py \
        "${DOCKER_BASE_DIR}" \
        "${MODEL_CONFIG}" \
        "${MODEL_CHECKPOINT}" \
        "--output-json" "${DOCKER_RESULTS_JSON}"

    # --- Step 7: Evaluation ---
    echo ""
    echo "--- Step 7: Evaluating results ---"
    # Convert Docker path to host path
    HOST_RESULTS_JSON="${INPUT_BIN_DIR}/predictions_${ANGLE}_${TIMESTAMP}.json"
    echo "Results JSON: ${HOST_RESULTS_JSON}"

    # Create evaluation results file
    EVAL_RESULTS_FILE="${BASE_DIR}/evaluation_results_angle_${ANGLE}_${TIMESTAMP}.txt"

    # Write header to results file
    echo "========================================================" > "${EVAL_RESULTS_FILE}"
    echo "KITTI Evaluation Results - Angle: ${ANGLE}" >> "${EVAL_RESULTS_FILE}"
    echo "Timestamp: ${TIMESTAMP}" >> "${EVAL_RESULTS_FILE}"
    echo "Input: ${INPUT_BIN_DIR}" >> "${EVAL_RESULTS_FILE}"
    echo "========================================================" >> "${EVAL_RESULTS_FILE}"
    echo "" >> "${EVAL_RESULTS_FILE}"

    uv run python kitti_eval_package/calculate_map_kitti_official.py \
        "${HOST_RESULTS_JSON}" \
        --gt-label-dir /data2/yoshida/label_kitti/training/label_2 | tee -a "${EVAL_RESULTS_FILE}"

    echo ""
    echo "========================================================" | tee -a "${EVAL_RESULTS_FILE}"
    echo "Evaluation results saved to: ${EVAL_RESULTS_FILE}" | tee -a "${EVAL_RESULTS_FILE}"
    echo "========================================================" | tee -a "${EVAL_RESULTS_FILE}"

    echo ""
    echo "Completed processing for ANGLE: ${ANGLE}"
    echo "=================================================="
done

echo ""
echo "=================================================="
echo "All angle configurations processed successfully!"
echo "=================================================="
