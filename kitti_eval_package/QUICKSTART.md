# Quick Start Guide

## Installation

1. Ensure you have `uv` installed (Python package manager)
2. Install required dependencies:
   ```bash
   uv pip install numpy
   ```

## Basic Usage

### Step 1: Prepare Your Files

You need:
1. **Prediction JSON file**: Your model's detection results
2. **Ground truth directory**: KITTI label files (`.txt` format)

### Step 2: Run Evaluation

```bash
uv run python calculate_map_kitti_official.py <your_predictions.json> \
    --gt-label-dir <path_to_kitti_labels>
```

### Example

```bash
uv run python calculate_map_kitti_official.py predictions.json \
    --gt-label-dir D:/label_kitti/training/label_2
```

### Step 3: Check Results

The script will:
1. Print results to console
2. Save detailed results to `<your_predictions>_map_official.txt`

## Expected Output

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

Results saved to: predictions_map_official.txt
```

## Common Options

- Change target class (default: Car):
  ```bash
  --class Pedestrian
  ```

- Change IoU threshold (default: 0.7 for Car):
  ```bash
  --min-overlap 0.5
  ```

## Troubleshooting

### "GT file not found" error
- Check that your ground truth directory path is correct
- Ensure GT files are named `<sample_token>.txt` (matching your prediction JSON)

### Import errors
- Make sure all 5 Python files are in the same directory:
  - `calculate_map_kitti_official.py`
  - `kitti_eval.py`
  - `kitti_label_parser.py`
  - `evaluate_kitti_predictions.py`
  - `coordinate_transform.py`

### Low AP scores
- Check that your predictions are in the correct coordinate system (Velodyne/LiDAR)
- Verify prediction JSON format matches the expected format (see README.md)
- Check that class names are correctly mapped (e.g., "vehicle.car" → "Car")

## Next Steps

See `README.md` for:
- Detailed file format specifications
- Implementation details
- Difficulty level criteria
- Complete API documentation
