#!/bin/bash

set -e  # Exit on error

ANGLES=(0_2 0_8 1 2 5 11 22 45)
BASE_DIR="/data2/yoshida/kitti_100"

echo "=================================================="
echo "Converting baseline denoised bl2 files to bin"
echo "Processing ${#ANGLES[@]} angle configurations"
echo "=================================================="

# Loop through each angle configuration
for ANGLE in "${ANGLES[@]}"; do
    echo ""
    echo "=================================================="
    echo "Processing ANGLE: ${ANGLE}"
    echo "=================================================="

    INPUT_DIR="${BASE_DIR}/baseline_denoised/${ANGLE}"
    OUTPUT_BIN_DIR="${BASE_DIR}/baseline_denoised_bin/${ANGLE}"

    # Check if input directory exists
    if [ ! -d "${INPUT_DIR}" ]; then
        echo "Warning: Input directory does not exist: ${INPUT_DIR}"
        echo "Skipping ANGLE: ${ANGLE}"
        continue
    fi

    echo "Converting bl2 to bin..."
    echo "  Input:  ${INPUT_DIR}"
    echo "  Output: ${OUTPUT_BIN_DIR}"

    uv run python datasets_generator/bl2_to_bin_converter.py \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_BIN_DIR}" \
        --amplitude-to-intensity-ratio 0.08 \
        --format kitti

    echo "Completed processing for ANGLE: ${ANGLE}"
    echo "=================================================="
done

echo ""
echo "=================================================="
echo "All angle configurations processed successfully!"
echo "=================================================="
