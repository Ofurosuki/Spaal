#!/bin/bash

# Description: Generates a large dataset by running the hist_matrix_generator.py script in batches.
#
# Usage: ./generate_dataset_batches.sh <total_frames> <batch_size> <pcd_directory> <output_dir> [other_python_script_options]
# Example: ./generate_dataset_batches.sh 1000 200 ./nuscenes_data/ ./lidar_datasets --lidar-type PCD_VLP32c

set -e # Exit immediately if a command exits with a non-zero status.

# --- Argument Parsing ---
# if [ "$#" -lt 4 ]; then
#   echo "Usage: $0 <total_frames> <batch_size> <pcd_directory> <output_dir> [other_python_script_options]"
#   echo "Example: $0 1000 200 ./nuscenes_data/ ./lidar_datasets --lidar-type PCD_VLP32c"
#   exit 1
# fi

# TOTAL_FRAMES=$1
# BATCH_SIZE=$2
# PCD_DIR=$3
# OUTPUT_DIR=$4
# shift 4 # Remove the first four arguments, leaving only the other options.
# OTHER_ARGS="$@"

TOTAL_FRAMES=1000
BATCH_SIZE=200
PCD_DIR="/data2/yoshida/pcd_1000"
OUTPUT_DIR="/data2/yoshida/hist_matrix_trainval_1"

# --- Main Loop ---
echo "Starting dataset generation..."
echo "Total frames:   $TOTAL_FRAMES"
echo "Batch size:     $BATCH_SIZE"
echo "PCD directory:  $PCD_DIR"
echo "Output directory: $OUTPUT_DIR"
#echo "Other options:  $OTHER_ARGS"
echo "================================================================"

for (( START_FRAME=0; START_FRAME<TOTAL_FRAMES; START_FRAME+=BATCH_SIZE )); do
  NUM_FRAMES=$BATCH_SIZE
  
  REMAINING_FRAMES=$((TOTAL_FRAMES - START_FRAME))
  if (( REMAINING_FRAMES < BATCH_SIZE )); then
    NUM_FRAMES=$REMAINING_FRAMES
  fi

  # The python script will append batch-specific info to this filename.
  FILENAME_PREFIX="lidar_signal"

  echo "----------------------------------------------------------------"
  echo "Generating Batch: Frames $START_FRAME to $((START_FRAME + NUM_FRAMES - 1))"
  echo "----------------------------------------------------------------"

  uv run python datasets_generator/hist_matrix_generator.py \
    --lidar-type PCD_VLP32c \
    --pcd-directory "$PCD_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --start-frame "$START_FRAME" \
    --num-frames "$NUM_FRAMES" \
    --output-filename "$FILENAME_PREFIX" \
    --spoofer-angle 90 \
    --spoofer-altitude 50

done

echo "================================================================"
echo "All batches generated successfully."
echo "================================================================"
