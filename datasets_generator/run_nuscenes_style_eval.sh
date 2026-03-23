#!/bin/bash

set -e  # Exit on error

ANGLES=(0_2 0_8 1 2 5 11 22 45)
BASE_DIR="/data2/yoshida"
DIR_SRC="kitti_100/swin_denoised_64_bin"

echo "=================================================="
echo "nuScenes-Style Evaluation from Latest Predictions"
echo "Source: ${DIR_SRC}"
echo "Angles: ${ANGLES[*]}"
echo "=================================================="

for ANGLE in "${ANGLES[@]}"; do
    echo ""
    echo "=================================================="
    echo "Processing ANGLE: ${ANGLE}"
    echo "=================================================="

    ANGLE_DIR="${BASE_DIR}/${DIR_SRC}/${ANGLE}"

    # --- PointPillars ---
    PP_JSON=$(ls -t "${ANGLE_DIR}"/predictions_pp_*.json 2>/dev/null | grep -v converted | head -1)

    if [ -z "${PP_JSON}" ]; then
        echo "  [PointPillars] No prediction JSON found, skipping."
    else
        echo "  [PointPillars] JSON: ${PP_JSON}"
        uv run python kitti_eval_package/calculate_map_nuscenes_style.py \
            "${PP_JSON}" \
            --gt-label-dir "${BASE_DIR}/label_kitti/training/label_2"
    fi

    # --- PV-RCNN (converted) ---
    PVRCNN_JSON=$(ls -t "${ANGLE_DIR}"/predictions_pvrcnn_*_converted.json 2>/dev/null | head -1)

    if [ -z "${PVRCNN_JSON}" ]; then
        echo "  [PV-RCNN] No converted prediction JSON found, skipping."
    else
        echo "  [PV-RCNN] JSON: ${PVRCNN_JSON}"
        uv run python kitti_eval_package/calculate_map_nuscenes_style.py \
            "${PVRCNN_JSON}" \
            --gt-label-dir "${BASE_DIR}/label_kitti/training/label_2"
    fi

    echo "Completed: ANGLE ${ANGLE}"
done

echo ""
echo "=================================================="
echo "All angles evaluated. Run summarize script:"
echo "  uv run python datasets_generator/summarize_eval_results_nuscenes_style.py"
echo "=================================================="
