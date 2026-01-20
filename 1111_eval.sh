#!/bin/bash

#JSON_PATH="/data2/yoshida/kitti_100/kitti_no_spoofer_bin/kitti_predictions_20251108_112225.json"
JSON_PATH="/data2/yoshida/kitti_100/baseline_denoised_bin/11/predictions_11_20251110_133318.json"
JSON_PATH="/data2/yoshida/kitti_100/nuscenes_bin/1/predictions_nuscenes_1_20251110_043459.json"
uv run python kitti_eval_package/calculate_map_nuscenes_style.py \
    "${JSON_PATH}" \
    --gt-label-dir /data2/yoshida/label_kitti/training/label_2 \
    --class Car \
    --score-threshold 0.0
 