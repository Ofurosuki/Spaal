#!/bin/bash

set -e  # Exit on error

ANGLES=(0_2 0_8 1 2 5 11 22 45)
BASE_DIR="/data2/yoshida"
#DIR_SRC="kitti_100/baseline_denoised_64_bin"
#DIR_SRC="kitti_100/baseline_bin_64"
DIR_SRC="kitti_100/kitti_bin_64"
# Docker configuration
DOCKER_CONTAINER_NAME="sharp_hodgkin"

# Model configurations
PP_CONFIG="configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
PP_CHECKPOINT="./nuscenes/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"

PVRCNN_CONFIG="configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py"
PVRCNN_CHECKPOINT="./nuscenes/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth"

echo "=================================================="
echo "Starting KITTI Inference & Evaluation Pipeline"
echo "Source: ${DIR_SRC}"
echo "Processing ${#ANGLES[@]} angle configurations: ${ANGLES[*]}"
echo "Models: PointPillars, PV-RCNN"
echo "=================================================="

for ANGLE in "${ANGLES[@]}"; do
    echo ""
    echo "=================================================="
    echo "Processing ANGLE: ${ANGLE}"
    echo "=================================================="

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    HOST_BIN_DIR="${BASE_DIR}/${DIR_SRC}/${ANGLE}"
    DOCKER_BIN_DIR="./nuscenes/${DIR_SRC}/${ANGLE}"
    EVAL_RESULTS_FILE="${HOST_BIN_DIR}/evaluation_results_${ANGLE}_${TIMESTAMP}.txt"

    echo "======================================================" > "${EVAL_RESULTS_FILE}"
    echo "KITTI Evaluation Results - Angle: ${ANGLE}" >> "${EVAL_RESULTS_FILE}"
    echo "Timestamp: ${TIMESTAMP}" >> "${EVAL_RESULTS_FILE}"
    echo "Source: ${HOST_BIN_DIR}" >> "${EVAL_RESULTS_FILE}"
    echo "======================================================" >> "${EVAL_RESULTS_FILE}"

    # --------------------------------------------------
    # PointPillars
    # --------------------------------------------------
    echo ""
    echo "--- [PointPillars] Step 1: Running inference ---"

    DOCKER_PRED_PP="${DOCKER_BIN_DIR}/predictions_pp_${ANGLE}_${TIMESTAMP}.json"
    HOST_PRED_PP="${HOST_BIN_DIR}/predictions_pp_${ANGLE}_${TIMESTAMP}.json"

    docker exec ${DOCKER_CONTAINER_NAME} python demo/my_inference.py \
        "${DOCKER_BIN_DIR}" \
        "${PP_CONFIG}" \
        "${PP_CHECKPOINT}" \
        "--output-json" "${DOCKER_PRED_PP}"

    echo ""
    echo "--- [PointPillars] Step 2: Evaluating results ---"

    echo "" >> "${EVAL_RESULTS_FILE}"
    echo "--- PointPillars Evaluation ---" >> "${EVAL_RESULTS_FILE}"
    echo "JSON: ${HOST_PRED_PP}" >> "${EVAL_RESULTS_FILE}"
    echo "" >> "${EVAL_RESULTS_FILE}"

    uv run python kitti_eval_package/calculate_map_nuscenes_style.py \
        "${HOST_PRED_PP}" \
        --gt-label-dir "${BASE_DIR}/label_kitti/training/label_2" | tee -a "${EVAL_RESULTS_FILE}"

    # --------------------------------------------------
    # PV-RCNN
    # --------------------------------------------------
    echo ""
    echo "--- [PV-RCNN] Step 1: Running inference ---"

    DOCKER_PRED_PVRCNN="${DOCKER_BIN_DIR}/predictions_pvrcnn_${ANGLE}_${TIMESTAMP}.json"
    HOST_PRED_PVRCNN="${HOST_BIN_DIR}/predictions_pvrcnn_${ANGLE}_${TIMESTAMP}.json"
    HOST_PRED_PVRCNN_CONV="${HOST_BIN_DIR}/predictions_pvrcnn_${ANGLE}_${TIMESTAMP}_converted.json"

    docker exec ${DOCKER_CONTAINER_NAME} python demo/my_inference.py \
        "${DOCKER_BIN_DIR}" \
        "${PVRCNN_CONFIG}" \
        "${PVRCNN_CHECKPOINT}" \
        "--output-json" "${DOCKER_PRED_PVRCNN}"

    echo ""
    echo "--- [PV-RCNN] Step 2: Converting class names ---"

    uv run python datasets_generator/convert_pvrcnn_classnames.py \
        "${HOST_PRED_PVRCNN}" \
        "${HOST_PRED_PVRCNN_CONV}"

    echo ""
    echo "--- [PV-RCNN] Step 3: Evaluating results ---"

    echo "" >> "${EVAL_RESULTS_FILE}"
    echo "--- PV-RCNN Evaluation ---" >> "${EVAL_RESULTS_FILE}"
    echo "JSON (converted): ${HOST_PRED_PVRCNN_CONV}" >> "${EVAL_RESULTS_FILE}"
    echo "" >> "${EVAL_RESULTS_FILE}"

    #uv run python kitti_eval_package/calculate_map_kitti_official.py \
    uv run python kitti_eval_package/calculate_map_nuscenes_style.py \
        "${HOST_PRED_PVRCNN_CONV}" \
        --gt-label-dir "${BASE_DIR}/label_kitti/training/label_2" | tee -a "${EVAL_RESULTS_FILE}"

    echo ""
    echo "======================================================" | tee -a "${EVAL_RESULTS_FILE}"
    echo "Evaluation results saved to: ${EVAL_RESULTS_FILE}" | tee -a "${EVAL_RESULTS_FILE}"
    echo "======================================================" | tee -a "${EVAL_RESULTS_FILE}"

    echo ""
    echo "Completed processing for ANGLE: ${ANGLE}"
    echo "=================================================="
done

echo ""
echo "=================================================="
echo "All angle configurations processed successfully!"
echo "=================================================="
