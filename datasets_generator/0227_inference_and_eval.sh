#!/bin/bash

set -e  # Exit on error

#ANGLES=(0_2 0_8 2 11 22 45)
ANGLES=(1)
BASE_DIR="/data2/yoshida"
DIR_SRC="swin_denoised_64_bin"

# Docker configuration
DOCKER_CONTAINER_NAME="sharp_hodgkin"
# MODEL_CONFIG="configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
# MODEL_CHECKPOINT="./nuscenes/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"

MODEL_CONFIG="configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py"
MODEL_CHECKPOINT="./nuscenes/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth"

# MODEL_CONFIG="configs/centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
# MODEL_CHECKPOINT="checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth"

echo "=================================================="
echo "Starting KITTI Inference & Evaluation Pipeline"
echo "Processing ${#ANGLES[@]} angle configurations"
echo "=================================================="

for ANGLE in "${ANGLES[@]}"; do
    echo ""
    echo "=================================================="
    echo "Processing ANGLE: ${ANGLE}"
    echo "=================================================="

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    EVAL_RESULTS_FILE="${BASE_DIR}/kitti_100/evaluation_results_angle_${ANGLE}_${TIMESTAMP}.txt"

    # # --- Step 5: Preprocess bin for CenterPoint (4xN -> 5xN, intensity * 255) ---
    # echo ""
    # echo "--- Step 5: Preprocessing bin for CenterPoint ---"

    # uv run python datasets_generator/preprocess_bin_for_centerpoint.py \
    #     "${HOST_BIN_DIR_SRC}" \
    #     "${HOST_BIN_DIR_CP}"

    # --- Step 6: Inference (CenterPoint) ---
    echo ""
    echo "--- Step 6: Running inference (CenterPoint) ---"
    #DOCKER_BASE_DIR_KITTI="./nuscenes/kitti_100/kitti_bin_64/${ANGLE}"
    DOCKER_BASE_DIR_KITTI="./nuscenes/kitti_100/${DIR_SRC}"
    DOCKER_RESULTS_JSON_KITTI="${DOCKER_BASE_DIR_KITTI}/predictions_kitti_${ANGLE}_${TIMESTAMP}.json"

    docker exec ${DOCKER_CONTAINER_NAME} python demo/my_inference.py \
        "${DOCKER_BASE_DIR_KITTI}" \
        "${MODEL_CONFIG}" \
        "${MODEL_CHECKPOINT}" \
        "--output-json" "${DOCKER_RESULTS_JSON_KITTI}"

    # --- Step 7: Convert PV-RCNN class names to KITTI class names ---
    echo ""
    echo "--- Step 7: Converting PV-RCNN class names to KITTI names ---"
    HOST_RESULTS_JSON_KITTI="${BASE_DIR}/kitti_100/${DIR_SRC}/predictions_kitti_${ANGLE}_${TIMESTAMP}.json"
    HOST_RESULTS_JSON_KITTI_CONVERTED="${BASE_DIR}/kitti_no_spoofer_bin_64_cp/predictions_kitti_${ANGLE}_${TIMESTAMP}_converted.json"

    uv run python datasets_generator/convert_pvrcnn_classnames.py \
        "${HOST_RESULTS_JSON_KITTI}" \
        "${HOST_RESULTS_JSON_KITTI_CONVERTED}"

    # --- Step 8: Evaluation (KITTI weights) ---
    echo ""
    echo "--- Step 8: Evaluating results (KITTI weights) ---"
    echo "Results JSON: ${HOST_RESULTS_JSON_KITTI_CONVERTED}"

    echo "========================================================" > "${EVAL_RESULTS_FILE}"
    echo "KITTI Evaluation Results - Angle: ${ANGLE}" >> "${EVAL_RESULTS_FILE}"
    echo "Timestamp: ${TIMESTAMP}" >> "${EVAL_RESULTS_FILE}"
    echo "========================================================" >> "${EVAL_RESULTS_FILE}"
    echo "" >> "${EVAL_RESULTS_FILE}"
    echo "--- KITTI Weights Evaluation ---" >> "${EVAL_RESULTS_FILE}"
    echo "JSON (converted): ${HOST_RESULTS_JSON_KITTI_CONVERTED}" >> "${EVAL_RESULTS_FILE}"
    echo "" >> "${EVAL_RESULTS_FILE}"

    uv run python kitti_eval_package/calculate_map_kitti_official.py \
        "${HOST_RESULTS_JSON_KITTI_CONVERTED}" \
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
