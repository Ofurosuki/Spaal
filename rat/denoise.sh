#!/bin/bash

# Batch Denoise and Inference Script
# 1. Uploads multiple signal.bl2 files.
# 2. Runs denoise on a remote server.
# 3. Converts denoised .bl2 to .pcd.bin on remote.
# 4. Runs CenterPoint inference on the .pcd.bin files.
# 5. Downloads all results (.bl2, .pcd.bin, .json) and organizes them.

# Stop on first error
set -e

# --- Configuration ---
# Denoise Config
REMOTE_USER="yoshida"
REMOTE_HOST="rat_server"
REMOTE_PROJECT_DIR="/home/yoshida/Spaal/HFR_Denoise"
REMOTE_BATCH_BASE_DIR="/data2/yoshida/denoise_batch"
CKPT_PATH="${REMOTE_PROJECT_DIR}/run/0923_dn_dim32_lat_attn_deg_45.pt"
MASK_EXPANSION=20

# Inference Config (from batch_inference_bin.sh)
DOCKER_CONTAINER_NAME="sharp_hodgkin"
REMOTE_INFERENCE_PROJECT_DIR="/home/yoshida/Spaal/mmdetection3d"
MODEL_CONFIG="configs/centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py" 
MODEL_CHECKPOINT="checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth"
# Assuming /data2 on host is accessible inside the Docker container
# The inference script will need absolute paths on the host.

# --- Script Usage ---
show_help() {
    echo "Usage: $0 --root-dir <dir> --scene <scene_name>"
    echo ""
    echo "Processes all 'signal.bl2' files found under '<root-dir>/<scene_name>/attacked/(id)/',"
    echo "then runs inference on the results."
    echo ""
    echo "Options:"
    echo "  --root-dir <dir>          Local root directory (e.g., 'D:/cvpr2026_data/0127')."
    echo "  --scene <scene_name>      The specific scene directory to process (e.g., 'scene_01')."
    echo "  -h, --help                Show this help message."
}

# --- Argument Parsing ---
ROOT_DIR=""
SCENE_NAME=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --root-dir) ROOT_DIR="$2"; shift 2;;
        --scene) SCENE_NAME="$2"; shift 2;;
        -h|--help) show_help; exit 0;;
        *) echo "Unknown option: $1"; show_help; exit 1;;
    esac
done

# --- Validation ---
if [ -z "${ROOT_DIR}" ] || [ -z "${SCENE_NAME}" ]; then
    echo "Error: Both --root-dir and --scene are required arguments."; show_help; exit 1;
fi
LOCAL_SCENE_DIR="${ROOT_DIR}/${SCENE_NAME}"
LOCAL_ATTACKED_DIR="${LOCAL_SCENE_DIR}/attacked"
LOCAL_DENOISED_DIR="${LOCAL_SCENE_DIR}/denoised"
if [ ! -d "${LOCAL_ATTACKED_DIR}" ]; then
    echo "Error: Source directory not found: ${LOCAL_ATTACKED_DIR}"; exit 1;
fi
BL2_FILES=$(find "${LOCAL_ATTACKED_DIR}" -type f -name "signal.bl2")
if [ -z "${BL2_FILES}" ]; then
    echo "Error: No 'signal.bl2' files found in subdirectories of ${LOCAL_ATTACKED_DIR}"; exit 1;
fi
NUM_FILES=$(echo "${BL2_FILES}" | wc -l)

echo "=========================================="
echo "Batch Denoise and Inference Pipeline"
echo "=========================================="
echo "Root directory:   ${ROOT_DIR}"
echo "Scene:            ${SCENE_NAME}"
echo "Files to process: ${NUM_FILES}"
echo "=========================================="

# --- Step 0: Setup remote directories ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REMOTE_INPUT_DIR="${REMOTE_BATCH_BASE_DIR}/input_${SCENE_NAME}_${TIMESTAMP}"
REMOTE_DENOISED_BL2_DIR="${REMOTE_BATCH_BASE_DIR}/denoised_bl2_${SCENE_NAME}_${TIMESTAMP}"
REMOTE_PCD_DIR="${REMOTE_BATCH_BASE_DIR}/pcd_${SCENE_NAME}_${TIMESTAMP}"
REMOTE_RESULTS_JSON="/data2/yoshida/inference_results/predictions_${SCENE_NAME}_${TIMESTAMP}.json"

echo -e "\n>>> Step 0: Setting up remote directories..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_INPUT_DIR} && mkdir -p ${REMOTE_DENOISED_BL2_DIR} && mkdir -p ${REMOTE_PCD_DIR}"
echo "✓ Remote directories created."

# --- Step 1: Upload all signal.bl2 files ---
echo -e "\n>>> Step 1: Uploading ${NUM_FILES} files..."
find "${LOCAL_ATTACKED_DIR}" -type f -name "signal.bl2" | while read -r FILE_PATH; do
    ID=$(basename "$(dirname "${FILE_PATH}")")
    REMOTE_FILENAME="signal_${ID}.bl2"
    scp "${FILE_PATH}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_INPUT_DIR}/${REMOTE_FILENAME}"
done
echo "✓ All files uploaded."

# --- Step 2: Run Denoise Pipeline ---
echo -e "\n>>> Step 2: Running denoise pipeline on remote server..."
REMOTE_CMD_DENOISE="cd ${REMOTE_PROJECT_DIR}/../ && \
    /home/yoshida/.local/bin/uv run python HFR_Denoise/pipeline/denoise_pipeline.py \
    --input-path ${REMOTE_INPUT_DIR} \
    --output-path ${REMOTE_DENOISED_BL2_DIR} \
    --ckpt-path ${CKPT_PATH} \
    --mask-expansion ${MASK_EXPANSION}"
ssh "${REMOTE_USER}@${REMOTE_HOST}" "${REMOTE_CMD_DENOISE}"
echo "✓ Denoise pipeline finished."

# --- Step 3: Convert .bl2 to .pcd.bin on remote ---
echo -e "\n>>> Step 3: Converting .bl2 to .pcd.bin on remote..."
REMOTE_CMD_CONVERT="cd ${REMOTE_PROJECT_DIR}/../ && \
    /home/yoshida/.local/bin/uv run python /home/yoshida/Spaal/bl2_to_bin_tools/convert_trimmed_bl2_to_pcd_bin.py ${REMOTE_DENOISED_BL2_DIR} ${REMOTE_PCD_DIR}"
ssh "${REMOTE_USER}@${REMOTE_HOST}" "${REMOTE_CMD_CONVERT}"
echo "✓ Conversion complete."

# --- Step 4: Run Inference on remote ---
echo -e "
>>> Step 4: Running inference inside Docker..."

# Convert the host path to the path visible inside the Docker container.
# Based on the info that host'/data2/yoshida' is mapped to what looks like './nuscenes' from the container's workdir.
# Assuming the container's workdir is /app, this means host'/data2/yoshida' maps to '/app/nuscenes'.
HOST_PATH_PREFIX="/data2/yoshida"
CONTAINER_PATH_PREFIX="./nuscenes" # Equivalent path inside the container

DOCKER_PCD_DIR=${REMOTE_PCD_DIR/#$HOST_PATH_PREFIX/$CONTAINER_PATH_PREFIX}
DOCKER_RESULTS_JSON=${REMOTE_RESULTS_JSON/#$HOST_PATH_PREFIX/$CONTAINER_PATH_PREFIX}

echo "  Host PCD path:     ${REMOTE_PCD_DIR}"
echo "  Container PCD path:  ${DOCKER_PCD_DIR}"
echo "  Host JSON path:    ${REMOTE_RESULTS_JSON}"
echo "  Container JSON path: ${DOCKER_RESULTS_JSON}"

REMOTE_CMD_INFER="docker exec ${DOCKER_CONTAINER_NAME} python demo/my_inference.py \
    ${DOCKER_PCD_DIR} \
    ${MODEL_CONFIG} \
    ${MODEL_CHECKPOINT} \
    --output-json ${DOCKER_RESULTS_JSON}"

ssh "${REMOTE_USER}@${REMOTE_HOST}" "${REMOTE_CMD_INFER}"
echo "✓ Inference complete."

# --- Step 5: Download all results ---
echo -e "\n>>> Step 5: Downloading all results..."
TEMP_DOWNLOAD_DIR_BL2=$(mktemp -d)
scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DENOISED_BL2_DIR}/*" "${TEMP_DOWNLOAD_DIR_BL2}"
TEMP_DOWNLOAD_DIR_PCD=$(mktemp -d)
scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PCD_DIR}/*" "${TEMP_DOWNLOAD_DIR_PCD}"
LOCAL_JSON_PATH="${LOCAL_SCENE_DIR}/predictions_${TIMESTAMP}.json"
LOG_FILE_PATH="${LOCAL_SCENE_DIR}/predictions_${TIMESTAMP}.txt"
scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_RESULTS_JSON}" "${LOCAL_JSON_PATH}"
echo "✓ All results downloaded. JSON saved to ${LOCAL_JSON_PATH}"

# --- Step 6: Organize downloaded files ---
echo -e "\n>>> Step 6: Organizing downloaded files..."
mkdir -p "${LOCAL_DENOISED_DIR}"

# The structure downloaded for .bl2 files is assumed to be 'signal_{id}/signal.bl2'.
# We loop through the ID directories found in the bl2 temp download dir.
for ID_DIR_PATH in "${TEMP_DOWNLOAD_DIR_BL2}"/*; do
    if [ -d "${ID_DIR_PATH}" ]; then
        # Get the directory name, which is assumed to be 'signal_{id}'
        ID_WITH_PREFIX=$(basename "${ID_DIR_PATH}")
        # Remove 'signal_' prefix to get the clean ID for the local target directory
        ID=${ID_WITH_PREFIX#signal_}

        TARGET_DIR="${LOCAL_DENOISED_DIR}/${ID}"
        mkdir -p "${TARGET_DIR}"

        # --- Organize .bl2 file ---
        BL2_FILE="${ID_DIR_PATH}/signal.bl2"
        if [ -f "${BL2_FILE}" ]; then
            echo "  Organizing .bl2 for ID ${ID} -> ${TARGET_DIR}/signal.bl2"
            mv "${BL2_FILE}" "${TARGET_DIR}/signal.bl2"
        else
            echo "  Warning: signal.bl2 not found for ID ${ID} in bl2 temp dir."
        fi

        # --- Organize .pcd.bin file ---
        # The .pcd.bin filename is likely based on the ID *with* the 'signal_' prefix.
        PCD_FILENAME="${ID_WITH_PREFIX}.pcd.bin"
PCD_FILE="${TEMP_DOWNLOAD_DIR_PCD}/${PCD_FILENAME}"
        if [ -f "${PCD_FILE}" ]; then
            echo "  Organizing ${PCD_FILENAME} -> ${TARGET_DIR}/signal.pcd.bin"
            mv "${PCD_FILE}" "${TARGET_DIR}/signal.pcd.bin"
        else
            echo "  Warning: Corresponding .pcd.bin file ('${PCD_FILENAME}') not found."
        fi
    fi
done
echo "✓ Files organized in ${LOCAL_DENOISED_DIR}"

# --- Step 7: Clean up ---
echo -e "\n>>> Step 7: Cleaning up temporary directories..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "rm -rf ${REMOTE_INPUT_DIR} ${REMOTE_DENOISED_BL2_DIR} ${REMOTE_PCD_DIR} ${REMOTE_RESULTS_JSON}"
echo "  ✓ Remote directories cleaned."
rm -rf "${TEMP_DOWNLOAD_DIR_BL2}" "${TEMP_DOWNLOAD_DIR_PCD}"
echo "  ✓ Local temporary directories cleaned."

# --- Step 8 (NEW): Record Experiment Details ---
echo -e "\n>>> Step 8: Recording experiment details to ${LOG_FILE_PATH}"
(
    echo "=========================================="
    echo "Batch Denoise and Inference Details"
    echo "=========================================="
    echo "Timestamp:          ${TIMESTAMP}"
    echo "Scene:              ${SCENE_NAME}"
    echo "Root Directory:     ${ROOT_DIR}"
    echo ""
    echo "--- Remote Paths (Temporary) ---"
    echo "Input .bl2 Dir:     ${REMOTE_INPUT_DIR}"
    echo "Denoised .bl2 Dir:  ${REMOTE_DENOISED_BL2_DIR}"
    echo "Converted .pcd Dir: ${REMOTE_PCD_DIR}"
    echo "Remote JSON Path:   ${REMOTE_RESULTS_JSON}"
    echo ""
    echo "--- Local Paths ---"
    echo "Final JSON Path:    ${LOCAL_JSON_PATH}"
    echo ""
    echo "--- Models ---"
    echo "Denoise Model:      ${CKPT_PATH}"
    echo "Inference Config:   ${MODEL_CONFIG}"
    echo "Inference Ckpt:     ${MODEL_CHECKPOINT}"
    echo "=========================================="
) | tee "${LOG_FILE_PATH}"
echo "✓ Details saved."


echo -e "\n=========================================="
echo "Pipeline finished successfully!"
echo "=========================================="
