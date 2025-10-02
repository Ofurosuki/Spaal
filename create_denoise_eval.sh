#!/bin/bash

# make sure execcute this script from the root directory where denoise_and_reconstruct_pipeline.py is located
set -e

MODEL_PATH="HFR_Denoise/run/0925_dn_dim32_lat_attn_deg_22.pt"
SYNC_ANGLE=22

CONTAINER_NAME="sharp_hodgkin"

echo "Start denoising and reconstruction..."
python denoise_and_reconstruct_pipeline.py --pcd-directory /data2/yoshida/minival_pcd/pcd_ascii --output-dir /data2/yoshida/1002_deg_"${SYNC_ANGLE}" --ckpt-path ./"${MODEL_PATH}" \
    --sync-angle-step-deg "${SYNC_ANGLE}"

echo "Start inference..."
docker exec -it "${CONTAINER_NAME}" python demo/my_inference.py ./nuscenes/1002_deg_"${SYNC_ANGLE}"/pcd_bin \
    configs/centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth \
    --output-json ./nuscenes/1002_deg_"${SYNC_ANGLE}".json

echo "Start evaluation..."
python evaluator.py /data2/yoshida/1002_deg_"${SYNC_ANGLE}".json --score-threshold 0.3 --fov-center -90 --fov-width 90