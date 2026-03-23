#!/bin/bash

uv run python datasets_generator/bl2_to_bin_converter.py \
    --input-dir /data2/yoshida/kitti_100/kitti_denoised_64/horizontal/1 \
    --output-dir /data2/yoshida/kitti_100/kitti_bin_64/1 \
    --amplitude-to-intensity-ratio 0.08 \
    --format kitti \
    --min-peak-amplitude 0.3