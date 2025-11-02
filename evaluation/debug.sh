#!/bin/bash
GT_DIR="D:/eval_cvpr2026/data/gt_bl2"
DENOISED_DIR="D:/eval_cvpr2026/data/denoised_bl2/sync_1"
#DENOISED_DIR="D:/eval_cvpr2026/data/denoised_baseline_bl2/sync_45"
THRESHOLD=0.5  # meters
MIN_GT_DISTANCE=0.0  # meters
EVAL_ANGLE=0  # 0 degrees = front (user-facing coordinate system)
EVAL_WIDTH=90  # 90 degrees width (matching spoofer-width-deg
OUTPUT_DIR="D:/eval_cvpr2026/evaluation_results/debug_sync_1"

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
    --max-samples 1\