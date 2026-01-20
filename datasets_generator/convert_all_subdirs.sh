#!/bin/bash

# Batch BL2 to BIN conversion script for multiple subdirectories
# Processes each subdirectory separately and saves results to corresponding output subdirectories

# Stop on first error
set -e

# --- Script Usage ---
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --parent-dir <dir>           Parent directory containing subdirectories with bl2 files (required)"
    echo "  --output-parent-dir <dir>    Parent output directory (required)"
    echo "  --amplitude-ratio <float>    Amplitude to intensity ratio (default: 1.3)"
    echo "  --use-answer-matrix          Use answer_matrix.bl2 instead of signal.bl2"
    echo "  --format <format>            Output format: 'nuscenes' or 'kitti' (default: nuscenes)"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --parent-dir D:/sony_dump_1026/hfr_bl2 \\"
    echo "     --output-parent-dir D:/eval_cvpr2026/data/converted"
    echo ""
    echo "This will process all subdirectories in the parent directory:"
    echo "  D:/sony_dump_1026/hfr_bl2/gt/       → D:/eval_cvpr2026/data/converted/gt/"
    echo "  D:/sony_dump_1026/hfr_bl2/attacked/ → D:/eval_cvpr2026/data/converted/attacked/"
    echo "  D:/sony_dump_1026/hfr_bl2/denoised/ → D:/eval_cvpr2026/data/converted/denoised/"
}

# --- Argument Parsing ---
PARENT_DIR=""
OUTPUT_PARENT_DIR=""
AMPLITUDE_RATIO="0.08"
USE_ANSWER_MATRIX=""
FORMAT="kitti"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --parent-dir)
            PARENT_DIR="$2"
            shift 2
            ;;
        --output-parent-dir)
            OUTPUT_PARENT_DIR="$2"
            shift 2
            ;;
        --amplitude-ratio)
            AMPLITUDE_RATIO="$2"
            shift 2
            ;;
        --use-answer-matrix)
            USE_ANSWER_MATRIX="--use-answer-matrix"
            shift
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# --- Validation ---
if [ -z "${PARENT_DIR}" ]; then
    echo "Error: --parent-dir is required"
    show_help
    exit 1
fi

if [ -z "${OUTPUT_PARENT_DIR}" ]; then
    echo "Error: --output-parent-dir is required"
    show_help
    exit 1
fi

if [ ! -d "${PARENT_DIR}" ]; then
    echo "Error: Parent directory not found: ${PARENT_DIR}"
    exit 1
fi

# Get list of subdirectories
SUBDIRS=()
for dir in "${PARENT_DIR}"/*/ ; do
    [ -d "$dir" ] || continue
    SUBDIRS+=("${dir%/}")
done

if [ ${#SUBDIRS[@]} -eq 0 ]; then
    echo "Error: No subdirectories found in ${PARENT_DIR}"
    exit 1
fi

echo "=========================================="
echo "Batch BL2 to BIN Conversion"
echo "=========================================="
echo "Parent directory: ${PARENT_DIR}"
echo "Output parent directory: ${OUTPUT_PARENT_DIR}"
echo "Found ${#SUBDIRS[@]} subdirectories:"
for subdir in "${SUBDIRS[@]}"; do
    echo "  - $(basename "${subdir}")"
done
echo "Amplitude ratio: ${AMPLITUDE_RATIO}"
echo "Use answer matrix: ${USE_ANSWER_MATRIX:-No}"
echo "Format: ${FORMAT}"
echo "=========================================="
echo ""

# Create output parent directory
mkdir -p "${OUTPUT_PARENT_DIR}"

# Process each subdirectory
SUCCESS_COUNT=0
FAIL_COUNT=0

for SUBDIR in "${SUBDIRS[@]}"; do
    SUBDIR_NAME=$(basename "${SUBDIR}")

    echo ""
    echo "######################################"
    echo "# Processing: ${SUBDIR_NAME}"
    echo "######################################"
    echo ""

    # Define output directory
    OUTPUT_DIR="${OUTPUT_PARENT_DIR}/${SUBDIR_NAME}"

    echo "Input:  ${SUBDIR}"
    echo "Output: ${OUTPUT_DIR}"
    echo ""

    # Check if subdirectory contains bl2 files or sample directories
    # Try to detect if it's a flat structure or folder structure
    BL2_COUNT=$(find "${SUBDIR}" -maxdepth 1 -name "*.bl2" 2>/dev/null | wc -l)
    SAMPLE_DIRS_COUNT=$(find "${SUBDIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)

    if [ ${BL2_COUNT} -gt 0 ]; then
        # Flat structure detected
        echo "Detected flat structure (${BL2_COUNT} .bl2 files)"
        CONVERTER_SCRIPT="data_generator/bl2_to_bin_converter_v2.py"
        EXTRA_ARGS="--structure flat"
    elif [ ${SAMPLE_DIRS_COUNT} -gt 0 ]; then
        # Folder structure detected
        echo "Detected folder structure (${SAMPLE_DIRS_COUNT} sample directories)"
        CONVERTER_SCRIPT="data_generator/bl2_to_bin_converter.py"
        EXTRA_ARGS=""
    else
        echo "Warning: No .bl2 files or sample directories found in ${SUBDIR}, skipping..."
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    echo "Using converter: ${CONVERTER_SCRIPT}"
    echo ""

    # Run conversion
    if uv run python "${CONVERTER_SCRIPT}" \
        --input-dir "${SUBDIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --amplitude-to-intensity-ratio ${AMPLITUDE_RATIO} \
        ${USE_ANSWER_MATRIX} \
        --format ${FORMAT} \
        ${EXTRA_ARGS}; then

        echo ""
        echo "✓ ${SUBDIR_NAME} conversion complete!"
        echo "========================================"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ""
        echo "✗ ${SUBDIR_NAME} conversion failed!"
        echo "========================================"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# Final summary
echo ""
echo "=========================================="
echo "BATCH CONVERSION COMPLETE"
echo "=========================================="
echo "Processed subdirectories: ${#SUBDIRS[@]}"
echo "Successful: ${SUCCESS_COUNT}"
echo "Failed: ${FAIL_COUNT}"
echo "Output directory: ${OUTPUT_PARENT_DIR}"
echo ""

if [ ${SUCCESS_COUNT} -gt 0 ]; then
    echo "Generated output directories:"
    for subdir in "${SUBDIRS[@]}"; do
        subdir_name=$(basename "${subdir}")
        output_dir="${OUTPUT_PARENT_DIR}/${subdir_name}"
        if [ -d "${output_dir}" ]; then
            bin_count=$(ls -1 "${output_dir}"/*.bin 2>/dev/null | wc -l)
            echo "  ${subdir_name}: ${bin_count} .bin files"
        fi
    done
fi

echo "=========================================="
echo ""

exit 0
