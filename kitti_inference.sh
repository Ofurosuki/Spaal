#!/bin/bash
REMOTE_USER="yoshida"
REMOTE_HOST="rat_server"
REMOTE_BASE_DIR="/data2/yoshida/sony"  # Host path (for scp)
DOCKER_CONTAINER_NAME="sharp_hodgkin"

#MODEL_CONFIG="configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py"
MODEL_CONFIG="configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
#MODEL_CHECKPOINT="./nuscenes/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"
MODEL_CHECKPOINT="./nuscenes/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"

MODEL_CONFIG="configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
MODEL_CHECKPOINT="./nuscenes/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"

BASE_DIR="/data2/yoshida"
BASE_DIR_KITTI="${BASE_DIR}/kitti_100/kitti/horizontal"
ANGLE="1"
DOCKER_BASE_DIR="./nuscenes/kitti_100/kitti_bin_5_elements/${ANGLE}"       # Docker container internal path, containing .pcd.bin
OUTPUT_BIN_DIR_KITTI="${BASE_DIR}/kitti_100/kitti_bin_5_elements/${ANGLE}"
# uv run python datasets_generator/bl2_to_bin_converter.py \
#         --input-dir "${BASE_DIR_KITTI}/${ANGLE}" \
#         --output-dir "${OUTPUT_BIN_DIR_KITTI}" \
#         --amplitude-to-intensity-ratio 0.08 \
#         --format nuscenes
# Generate timestamp for output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DOCKER_RESULTS_JSON="${DOCKER_BASE_DIR}/kitti_predictions_${TIMESTAMP}.json"  # Docker path
docker exec ${DOCKER_CONTAINER_NAME} python demo/my_inference.py \
    "${DOCKER_BASE_DIR}" \
    "${MODEL_CONFIG}" \
    "${MODEL_CHECKPOINT}" \
    "--output-json" "${DOCKER_RESULTS_JSON}"
