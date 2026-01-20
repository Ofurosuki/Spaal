#!/bin/bash
INPUT_DIR="/data2/yoshida/kitti_100/attacked_64"
TEST_ANGLE="1"
BASE_DIR="/data2/yoshida/kitti_100"
BASE_DIR_NUSCENES="${BASE_DIR}/nuscenes_denoised_64/horizontal"
MASK_EXPANSION=5
DENOISE_CKPT_PATH_NUSCENES="HFR_Denoise/run/pretrain_general_data_dim32_attn.pt"
uv run python HFR_Denoise/pipeline/denoise_pipeline.py \
    --input-path "${INPUT_DIR}/${TEST_ANGLE}" \
    --output-path "${BASE_DIR_NUSCENES}/${TEST_ANGLE}" \
    --ckpt-path "${DENOISE_CKPT_PATH_NUSCENES}" \
    --mask-expansion "${MASK_EXPANSION}" \
    --chunk-size 20 \
    --split-h 2