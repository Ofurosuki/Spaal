#!/usr/bin/env python3
"""
Check the spatial distribution of predictions to see if they extend beyond
KITTI's evaluation range (forward direction only).
"""

import json
import sys
import numpy as np
from collections import defaultdict

def analyze_prediction_range(json_file):
    """Analyze spatial distribution of predictions."""

    print(f"Analyzing: {json_file}")
    print("=" * 80)

    with open(json_file, 'r') as f:
        data = json.load(f)

    results = data.get('results', {})

    # Collect statistics
    total_predictions = 0
    x_coords = []
    y_coords = []
    z_coords = []

    forward_count = 0  # x > 0
    backward_count = 0  # x < 0
    left_count = 0     # y > 0
    right_count = 0    # y < 0

    class_distribution = defaultdict(int)

    for sample_token, predictions in results.items():
        for pred in predictions:
            total_predictions += 1

            # Get translation [x, y, z] in KITTI LiDAR coordinates
            translation = pred['translation']
            x, y, z = translation

            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)

            # Count by direction
            if x > 0:
                forward_count += 1
            else:
                backward_count += 1

            if y > 0:
                left_count += 1
            else:
                right_count += 1

            # Class distribution
            class_name = pred.get('detection_name', 'unknown')
            class_distribution[class_name] += 1

    # Convert to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    z_coords = np.array(z_coords)

    # Print summary
    print(f"\nTotal predictions: {total_predictions}")
    print(f"Samples: {len(results)}")
    print(f"Avg predictions per sample: {total_predictions / len(results):.1f}")

    print(f"\n{'='*80}")
    print("SPATIAL DISTRIBUTION")
    print(f"{'='*80}")

    print(f"\nX-axis (forward/backward):")
    print(f"  Range: [{x_coords.min():.2f}, {x_coords.max():.2f}] meters")
    print(f"  Mean: {x_coords.mean():.2f} m")
    print(f"  Std: {x_coords.std():.2f} m")
    print(f"  Forward (x > 0): {forward_count} ({forward_count/total_predictions*100:.1f}%)")
    print(f"  Backward (x < 0): {backward_count} ({backward_count/total_predictions*100:.1f}%)")

    if backward_count > 0:
        print(f"\n⚠️  WARNING: {backward_count} predictions are in backward direction (x < 0)")
        print(f"   KITTI ground truth only covers forward direction!")
        print(f"   These should be filtered out before evaluation.")

        # Show some examples of backward predictions
        backward_samples = []
        for sample_token, predictions in results.items():
            for pred in predictions:
                x = pred['translation'][0]
                if x < 0:
                    backward_samples.append({
                        'sample': sample_token,
                        'x': x,
                        'y': pred['translation'][1],
                        'score': pred['detection_score'],
                        'class': pred.get('detection_name', 'unknown')
                    })
                    if len(backward_samples) >= 5:
                        break
            if len(backward_samples) >= 5:
                break

        print(f"\n   Examples of backward predictions:")
        for i, sample in enumerate(backward_samples, 1):
            print(f"     {i}. Sample {sample['sample']}: x={sample['x']:.2f}, y={sample['y']:.2f}, "
                  f"score={sample['score']:.3f}, class={sample['class']}")

    print(f"\nY-axis (left/right):")
    print(f"  Range: [{y_coords.min():.2f}, {y_coords.max():.2f}] meters")
    print(f"  Mean: {y_coords.mean():.2f} m")
    print(f"  Std: {y_coords.std():.2f} m")
    print(f"  Left (y > 0): {left_count} ({left_count/total_predictions*100:.1f}%)")
    print(f"  Right (y < 0): {right_count} ({right_count/total_predictions*100:.1f}%)")

    print(f"\nZ-axis (height):")
    print(f"  Range: [{z_coords.min():.2f}, {z_coords.max():.2f}] meters")
    print(f"  Mean: {z_coords.mean():.2f} m")
    print(f"  Std: {z_coords.std():.2f} m")

    print(f"\n{'='*80}")
    print("CLASS DISTRIBUTION")
    print(f"{'='*80}")
    for class_name, count in sorted(class_distribution.items(), key=lambda x: -x[1]):
        print(f"  {class_name:<30} {count:>6} ({count/total_predictions*100:>5.1f}%)")

    # KITTI evaluation range check
    print(f"\n{'='*80}")
    print("KITTI EVALUATION RANGE")
    print(f"{'='*80}")

    # KITTI typically evaluates in the forward hemisphere
    # Common range: x > 0, |y| < 50m, z in reasonable range
    kitti_range_x_min = 0.0
    kitti_range_x_max = 70.0  # Typical max range
    kitti_range_y = 40.0  # Typical lateral range

    in_range_count = 0
    for x, y in zip(x_coords, y_coords):
        if x > kitti_range_x_min and x < kitti_range_x_max and abs(y) < kitti_range_y:
            in_range_count += 1

    print(f"\nAssuming KITTI range:")
    print(f"  X: ({kitti_range_x_min}, {kitti_range_x_max}) meters (forward)")
    print(f"  Y: (-{kitti_range_y}, {kitti_range_y}) meters (lateral)")
    print(f"\nPredictions in KITTI range: {in_range_count} ({in_range_count/total_predictions*100:.1f}%)")
    print(f"Predictions outside KITTI range: {total_predictions - in_range_count} "
          f"({(total_predictions - in_range_count)/total_predictions*100:.1f}%)")

    if (total_predictions - in_range_count) > 0:
        print(f"\n⚠️  {(total_predictions - in_range_count)/total_predictions*100:.1f}% of predictions "
              f"are outside typical KITTI evaluation range.")
        print(f"   These will be counted as false positives if not filtered!")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_prediction_range.py <prediction_json>")
        sys.exit(1)

    analyze_prediction_range(sys.argv[1])
