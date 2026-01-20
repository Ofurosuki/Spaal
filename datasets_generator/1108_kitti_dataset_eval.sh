#!/bin/bash

set -e  # Exit on error

ANGLES=(0_8 2 5 11 22 45)
INPUT_DIR="/data2/yoshida/kitti_100/kitti/horizontal"

BASE_DIR="/data2/yoshida/kitti_100"
BASE_DIR_NUSCENES="${BASE_DIR}/nuscenes_denoised/horizontal"
BASE_DIR_KITTI="${BASE_DIR}/kitti_denoised/horizontal"
DENOISE_CKPT_PATH_NUSCENES="HFR_Denoise/run/pretrain_general_data_dim32_attn.pt"
DENOISE_CKPT_PATH_KITTI="HFR_Denoise/run/1030_dn_dim32_lat_attn_deg_0_kitti.pt"
MASK_EXPANSION=5

# Docker configuration
DOCKER_CONTAINER_NAME="sharp_hodgkin"  # Update this to match your container name
MODEL_CONFIG="configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
MODEL_CHECKPOINT="./nuscenes/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"

# Datasets used in this script are KITTI datasets.

echo "=================================================="
echo "Starting KITTI Dataset Evaluation Pipeline"
echo "Processing ${#ANGLES[@]} angle configurations"
echo "=================================================="

# Loop through each angle configuration
for ANGLE in "${ANGLES[@]}"; do
    echo ""
    echo "=================================================="
    echo "Processing ANGLE: ${ANGLE}"
    echo "=================================================="

    # --- 1. Denoising using model trained by nuScenes ---
    echo ""
    echo "--- Step 1: Denoising with nuScenes weights ---"
    uv run python HFR_Denoise/pipeline/denoise_pipeline.py \
        --input-path "${INPUT_DIR}/${ANGLE}" \
        --output-path "${BASE_DIR_NUSCENES}/${ANGLE}" \
        --ckpt-path "${DENOISE_CKPT_PATH_NUSCENES}" \
        --mask-expansion "${MASK_EXPANSION}"

    # --- 2. Denoising using model trained by KITTI ---
    echo ""
    echo "--- Step 2: Denoising with KITTI weights ---"
    uv run python HFR_Denoise/pipeline/denoise_pipeline.py \
        --input-path "${INPUT_DIR}/${ANGLE}" \
        --output-path "${BASE_DIR_KITTI}/${ANGLE}" \
        --ckpt-path "${DENOISE_CKPT_PATH_KITTI}" \
        --mask-expansion "${MASK_EXPANSION}"

    # --- 3. Convert bl2 to bin (nuScenes weights) ---
    echo ""
    echo "--- Step 3: Converting bl2 to bin (nuScenes weights) ---"
    OUTPUT_BIN_DIR_NUSCENES="${BASE_DIR}/nuscenes_bin/${ANGLE}"
    uv run python datasets_generator/bl2_to_bin_converter.py \
        --input-dir "${BASE_DIR_NUSCENES}/${ANGLE}" \
        --output-dir "${OUTPUT_BIN_DIR_NUSCENES}" \
        --amplitude-to-intensity-ratio 0.08 \
        --format kitti

    # --- 4. Convert bl2 to bin (KITTI weights) ---
    echo ""
    echo "--- Step 4: Converting bl2 to bin (KITTI weights) ---"
    OUTPUT_BIN_DIR_KITTI="${BASE_DIR}/kitti_bin/${ANGLE}"
    uv run python datasets_generator/bl2_to_bin_converter.py \
        --input-dir "${BASE_DIR_KITTI}/${ANGLE}" \
        --output-dir "${OUTPUT_BIN_DIR_KITTI}" \
        --amplitude-to-intensity-ratio 0.08 \
        --format kitti

    # --- 5. Inference (nuScenes weights) ---
    echo ""
    echo "--- Step 5: Running inference (nuScenes weights) ---"
    DOCKER_BASE_DIR_NUSCENES="./nuscenes/kitti_100/nuscenes_bin/${ANGLE}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    DOCKER_RESULTS_JSON_NUSCENES="${DOCKER_BASE_DIR_NUSCENES}/predictions_nuscenes_${ANGLE}_${TIMESTAMP}.json"

    docker exec ${DOCKER_CONTAINER_NAME} python demo/my_inference.py \
        "${DOCKER_BASE_DIR_NUSCENES}" \
        "${MODEL_CONFIG}" \
        "${MODEL_CHECKPOINT}" \
        "--output-json" "${DOCKER_RESULTS_JSON_NUSCENES}"

    # --- 6. Inference (KITTI weights) ---
    echo ""
    echo "--- Step 6: Running inference (KITTI weights) ---"
    DOCKER_BASE_DIR_KITTI="./nuscenes/kitti_100/kitti_bin/${ANGLE}"
    DOCKER_RESULTS_JSON_KITTI="${DOCKER_BASE_DIR_KITTI}/predictions_kitti_${ANGLE}_${TIMESTAMP}.json"

    docker exec ${DOCKER_CONTAINER_NAME} python demo/my_inference.py \
        "${DOCKER_BASE_DIR_KITTI}" \
        "${MODEL_CONFIG}" \
        "${MODEL_CHECKPOINT}" \
        "--output-json" "${DOCKER_RESULTS_JSON_KITTI}"

    # Create evaluation results file for this angle
    EVAL_RESULTS_FILE="${BASE_DIR}/evaluation_results_angle_${ANGLE}_${TIMESTAMP}.txt"

    # --- 7. Evaluation (nuScenes weights) ---
    echo ""
    echo "--- Step 7: Evaluating results (nuScenes weights) ---"
    # Convert Docker path to host path
    HOST_RESULTS_JSON_NUSCENES="/data2/yoshida/kitti_100/nuscenes_bin/${ANGLE}/predictions_nuscenes_${ANGLE}_${TIMESTAMP}.json"
    echo "Results JSON: ${HOST_RESULTS_JSON_NUSCENES}"

    # Write header to results file
    echo "========================================================" > "${EVAL_RESULTS_FILE}"
    echo "KITTI Evaluation Results - Angle: ${ANGLE}" >> "${EVAL_RESULTS_FILE}"
    echo "Timestamp: ${TIMESTAMP}" >> "${EVAL_RESULTS_FILE}"
    echo "========================================================" >> "${EVAL_RESULTS_FILE}"
    echo "" >> "${EVAL_RESULTS_FILE}"

    echo "--- nuScenes Weights Evaluation ---" >> "${EVAL_RESULTS_FILE}"
    echo "JSON: ${HOST_RESULTS_JSON_NUSCENES}" >> "${EVAL_RESULTS_FILE}"
    echo "" >> "${EVAL_RESULTS_FILE}"

    uv run python kitti_eval_package/calculate_map_kitti_official.py \
        "${HOST_RESULTS_JSON_NUSCENES}" \
        --gt-label-dir /data2/yoshida/label_kitti/training/label_2 | tee -a "${EVAL_RESULTS_FILE}"

    # --- 8. Evaluation (KITTI weights) ---
    echo ""
    echo "--- Step 8: Evaluating results (KITTI weights) ---"
    # Convert Docker path to host path
    HOST_RESULTS_JSON_KITTI="/data2/yoshida/kitti_100/kitti_bin/${ANGLE}/predictions_kitti_${ANGLE}_${TIMESTAMP}.json"
    echo "Results JSON: ${HOST_RESULTS_JSON_KITTI}"

    echo "" >> "${EVAL_RESULTS_FILE}"
    echo "========================================================" >> "${EVAL_RESULTS_FILE}"
    echo "--- KITTI Weights Evaluation ---" >> "${EVAL_RESULTS_FILE}"
    echo "JSON: ${HOST_RESULTS_JSON_KITTI}" >> "${EVAL_RESULTS_FILE}"
    echo "" >> "${EVAL_RESULTS_FILE}"

    uv run python kitti_eval_package/calculate_map_kitti_official.py \
        "${HOST_RESULTS_JSON_KITTI}" \
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
