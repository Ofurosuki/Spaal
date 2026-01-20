#!/bin/bash

# Test script for single angle configuration
set -e  # Exit on error

# Test with single angle
TEST_ANGLE="1"
INPUT_DIR="/data2/yoshida/kitti_100/kitti/horizontal"

BASE_DIR="/data2/yoshida/kitti_100"
BASE_DIR_NUSCENES="${BASE_DIR}/kitti_low_intensity_denoised"
BASE_DIR_KITTI="${BASE_DIR}/kitti_low_intensity_denoised"
DENOISE_CKPT_PATH_NUSCENES="HFR_Denoise/run/pretrain_general_data_dim32_attn.pt"
DENOISE_CKPT_PATH_KITTI="HFR_Denoise/run/1030_dn_dim32_lat_attn_deg_0_kitti.pt"
MASK_EXPANSION=5

# Docker configuration
DOCKER_CONTAINER_NAME="sharp_hodgkin"
MODEL_CONFIG="configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
MODEL_CHECKPOINT="./nuscenes/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"

echo "=================================================="
echo "Test run for ANGLE: ${TEST_ANGLE}"
echo "=================================================="

# # --- 1. Denoising using model trained by nuScenes ---
echo ""
echo "--- Step 1: Denoising with nuScenes weights ---"
# uv run python HFR_Denoise/pipeline/denoise_pipeline.py \
#     --input-path "${INPUT_DIR}" \
#     --output-path "${BASE_DIR_NUSCENES}/${TEST_ANGLE}" \
#     --ckpt-path "${DENOISE_CKPT_PATH_NUSCENES}" \
#     --mask-expansion "${MASK_EXPANSION}"

# # --- 2. Denoising using model trained by KITTI ---
# echo ""
# echo "--- Step 2: Denoising with KITTI weights ---"
uv run python HFR_Denoise/pipeline/denoise_pipeline.py \
    --input-path "${INPUT_DIR}/${TEST_ANGLE}" \
    --output-path "${BASE_DIR_KITTI}/${TEST_ANGLE}" \
    --ckpt-path "${DENOISE_CKPT_PATH_KITTI}" \
    --mask-expansion "${MASK_EXPANSION}"