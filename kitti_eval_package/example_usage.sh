#!/bin/bash
# Example usage script for KITTI mAP evaluation

# Basic usage (with default GT directory)
# uv run python calculate_map_kitti_official.py predictions.json

# With custom GT directory
# uv run python calculate_map_kitti_official.py predictions.json \
#     --gt-label-dir /path/to/kitti/training/label_2

# Evaluate Car class with IoU 0.7 (KITTI standard)
# uv run python calculate_map_kitti_official.py predictions.json \
#     --class Car \
#     --min-overlap 0.7

# Evaluate Pedestrian class with IoU 0.5
# uv run python calculate_map_kitti_official.py predictions.json \
#     --class Pedestrian \
#     --min-overlap 0.5

# Full example
uv run python calculate_map_kitti_official.py predictions.json \
    --gt-label-dir D:/label_kitti/training/label_2 \
    --class Car \
    --min-overlap 0.7
