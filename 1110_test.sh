#!/bin/bash
INPUT_DIR="./nuscenes/kitti_100/kitti_low_intensity_denoised_bin/1"

# Docker configuration
DOCKER_CONTAINER_NAME="sharp_hodgkin"
MODEL_CONFIG="configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
MODEL_CHECKPOINT="./nuscenes/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"
docker exec "${DOCKER_CONTAINER_NAME}" python demo/my_inference.py \
    "${INPUT_DIR}" \
    "${MODEL_CONFIG}" \
    "${MODEL_CHECKPOINT}" \
    --output-json ./nuscenes/kitti_100/kitti_low_intensity_denoised_bin/preds.json

echo "Inference complete!"

uv run python kitti_eval_package/calculate_map_kitti_official.py \
    /data2/yoshida/kitti_100/kitti_low_intensity_denoised_bin/preds.json \
    --gt-label-dir /data2/yoshida/label_kitti/training/label_2