# KITTI Official mAP Evaluation Package

This package calculates mAP (mean Average Precision) for KITTI 3D object detection predictions following the official KITTI C++ implementation.

## Features

- Follows KITTI official C++ implementation (`evaluate_object.cpp`)
- 41-point recall sampling (KITTI standard)
- Precision smoothing: `max_{i..end}(precision)`
- Evaluates by difficulty levels (Easy, Moderate, Hard)
- Supports Car class evaluation with IoU threshold 0.7

## Files

```
kitti_eval_package/
├── calculate_map_kitti_official.py  # Main evaluation script
├── kitti_eval.py                    # Data structures (BBox3D, Detection, GroundTruth)
├── kitti_label_parser.py            # KITTI label file parser
├── evaluate_kitti_predictions.py   # Prediction JSON parser and IoU calculation
├── coordinate_transform.py          # Camera ↔ Velodyne coordinate transformations
└── README.md                        # This file
```

## Requirements

Python packages (install via `uv` or `pip`):
- numpy
- (Standard library: json, sys, os, pathlib, typing, dataclasses, collections)

## Usage

### Basic Evaluation

```bash
uv run python calculate_map_kitti_official.py <prediction_json_file>
```

### With Custom GT Directory

```bash
uv run python calculate_map_kitti_official.py <prediction_json_file> \
    --gt-label-dir /path/to/kitti/labels
```

### Arguments

- `pred_file` (required): Path to prediction JSON file
- `--gt-label-dir`: Directory containing ground truth label files (default: `D:\label_kitti\training\label_2`)
- `--class`: Target class to evaluate (default: `Car`)
- `--min-overlap`: Minimum IoU threshold (default: `0.7` for Car)

### Example

```bash
uv run python calculate_map_kitti_official.py predictions.json \
    --gt-label-dir D:/label_kitti/training/label_2 \
    --class Car \
    --min-overlap 0.7
```

## Prediction JSON Format

The prediction file should follow this format:

```json
{
  "results": {
    "sample_token_1": [
      {
        "sample_token": "sample_token_1",
        "translation": [x, y, z],
        "size": [w, l, h],
        "rotation": [qw, qx, qy, qz],
        "velocity": [vx, vy],
        "detection_name": "vehicle.car",
        "detection_score": 0.95,
        "attribute_name": "vehicle.moving"
      }
    ],
    "sample_token_2": [...]
  }
}
```

**Notes:**
- `translation`: [x, y, z] in Velodyne (LiDAR) coordinates
- `size`: [width, length, height]
- `rotation`: Quaternion [w, x, y, z] (will be converted to yaw angle)
- `detection_name`: Class name (e.g., "vehicle.car" → "Car")
- `detection_score`: Confidence score (0.0 - 1.0)

## Ground Truth Format

Ground truth files should be KITTI label format (`.txt` files):

```
Car 0.00 0 -1.57 599.41 156.40 629.75 189.25 1.48 1.60 3.69 2.84 1.47 8.41 -1.56
```

Each line contains 15 values:
1. Class name (e.g., "Car", "Pedestrian", "Cyclist")
2. Truncation (0.0 - 1.0)
3. Occlusion (0, 1, 2, 3)
4. Alpha (observation angle)
5-8. 2D bounding box (left, top, right, bottom)
9-11. 3D dimensions (height, width, length)
12-14. 3D location (x, y, z in camera coordinates)
15. Rotation_y (yaw angle)

## Output

The script outputs:

1. **Console output**: Summary statistics and AP by difficulty
2. **Result file**: `<prediction_file>_map_official.txt` containing:
   - AP for ALL difficulties combined
   - AP for Easy, Moderate, Hard separately
   - Precision-recall curves for each difficulty level

### Example Output

```
================================================================================
RESULTS SUMMARY
================================================================================

Class: Car
Minimum overlap (IoU): 0.7

Difficulty      AP              AP%
---------------------------------------------
ALL             0.7414          74.14
EASY            0.5856          58.56
MODERATE        0.2693          26.93
HARD            0.0613          6.13
```

## Difficulty Levels (KITTI Standard)

Objects are classified into difficulty levels based on:
- **Occlusion**: 0 (fully visible), 1 (partly occluded), 2 (largely occluded), 3 (unknown)
- **Truncation**: Percentage of object outside image (0.0 - 1.0)
- **2D bounding box height**: Pixel height in image

### Car Class Criteria:
- **Easy**: Min. height 40px, Max. occlusion level 0 (fully visible), Max. truncation 15%
- **Moderate**: Min. height 25px, Max. occlusion level 1 (partly occluded), Max. truncation 30%
- **Hard**: Min. height 25px, Max. occlusion level 2 (largely occluded), Max. truncation 50%

## Implementation Details

This implementation follows the official KITTI C++ code:
- **Threshold selection**: Uses recall-based threshold discretization (41 points)
- **Precision smoothing**: `precision[i] = max(precision[i:])`
- **AP calculation**: Mean of precision values at 41 recall points
- **IoU calculation**: Bird's Eye View (BEV) IoU approximation using center distance
- **Coordinate systems**: Handles Camera ↔ Velodyne transformations

## Reference

Based on KITTI official evaluation code:
- `D:/kitti_evaluation/cpp/evaluate_object.cpp`
- KITTI 3D Object Detection Benchmark: http://www.cvlibs.net/datasets/kitti/eval_object.php

## Notes

- The "ALL" AP (all difficulties together) can be higher than individual difficulty APs because:
  - ALL evaluation uses broader score thresholds (more detections included)
  - Detections can match any GT regardless of difficulty
  - Individual difficulty evaluations restrict matching to that difficulty only

- Ground truth objects with `num_lidar_pts < 1` are excluded from evaluation

- Class names are automatically converted:
  - "vehicle.car" → "Car"
  - "human.pedestrian" → "Pedestrian"
  - "vehicle.bicycle" → "Cyclist"
