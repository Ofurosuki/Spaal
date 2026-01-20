#!/usr/bin/env python3
"""
Verify that predictions and ground truth are in the same coordinate system.
"""

import sys
import json
sys.path.append('/home/yoshida/Spaal/kitti_eval_package')
from kitti_label_parser import KITTILabelParser
import numpy as np

def verify_coordinates(pred_file, gt_label_dir, sample_id='000007'):
    """Verify coordinate system consistency."""

    print("=" * 80)
    print("COORDINATE SYSTEM VERIFICATION")
    print("=" * 80)

    # Load predictions
    with open(pred_file, 'r') as f:
        preds = json.load(f)['results'][sample_id]

    # Load GT in LiDAR coordinates
    gt_file = f'{gt_label_dir}/{sample_id}.txt'
    gts_lidar = KITTILabelParser.parse_ground_truth_file(gt_file, coordinate_system='lidar')
    gts_camera = KITTILabelParser.parse_ground_truth_file(gt_file, coordinate_system='camera')

    car_gts_lidar = [g for g in gts_lidar if g.class_name == 'Car']
    car_gts_camera = [g for g in gts_camera if g.class_name == 'Car']

    if not car_gts_lidar:
        print(f"No Car GT in sample {sample_id}")
        return False

    print(f"\nSample: {sample_id}")
    print(f"GT Cars: {len(car_gts_lidar)}")
    print(f"Predictions: {len(preds)}")

    # Compare first GT with closest prediction
    gt_l = car_gts_lidar[0]
    gt_c = car_gts_camera[0]

    print(f"\nFirst GT Car:")
    print(f"  Camera coords (x=right, y=down, z=forward):")
    print(f"    x={gt_c.bbox.x:7.2f}, y={gt_c.bbox.y:7.2f}, z={gt_c.bbox.z:7.2f}")
    print(f"  LiDAR coords  (x=forward, y=left, z=up):")
    print(f"    x={gt_l.bbox.x:7.2f}, y={gt_l.bbox.y:7.2f}, z={gt_l.bbox.z:7.2f}")

    # Find closest prediction
    min_dist_lidar = float('inf')
    min_dist_camera = float('inf')
    closest_pred = None

    for pred in preds:
        px, py, pz = pred['translation']

        # Distance in LiDAR frame
        dist_lidar = np.sqrt((px - gt_l.bbox.x)**2 + (py - gt_l.bbox.y)**2)

        # Distance in Camera frame
        dist_camera = np.sqrt((px - gt_c.bbox.x)**2 + (py - gt_c.bbox.y)**2)

        if dist_lidar < min_dist_lidar:
            min_dist_lidar = dist_lidar
            min_dist_camera_for_closest = dist_camera
            closest_pred = pred

    if closest_pred:
        px, py, pz = closest_pred['translation']
        print(f"\nClosest Prediction (score={closest_pred['detection_score']:.3f}):")
        print(f"  x={px:7.2f}, y={py:7.2f}, z={pz:7.2f}")

        print(f"\nDistance Analysis:")
        print(f"  If predictions are in LiDAR coords:")
        print(f"    BEV distance from GT: {min_dist_lidar:.2f}m  ← Should be small")
        print(f"  If predictions are in Camera coords:")
        print(f"    BEV distance from GT: {min_dist_camera_for_closest:.2f}m  ← Would be large")

        print(f"\n{'=' * 80}")
        if min_dist_lidar < 2.0:  # Reasonable match threshold
            print("✅ VERIFICATION PASSED")
            print("   Predictions are in KITTI LiDAR coordinate system")
            print("   (x=forward, y=left, z=up)")
            return True
        else:
            print("⚠️  VERIFICATION FAILED")
            print("   Coordinate system mismatch detected!")
            return False

    return False


if __name__ == "__main__":
    pred_file = '/data2/yoshida/kitti_100/kitti_no_spoofer_bin/kitti_predictions_20251108_112225.json'
    gt_label_dir = '/data2/yoshida/label_kitti/training/label_2'

    # Test with multiple samples
    test_samples = ['000007', '000180', '000311']

    results = []
    for sample_id in test_samples:
        try:
            result = verify_coordinates(pred_file, gt_label_dir, sample_id)
            results.append((sample_id, result))
            print()
        except Exception as e:
            print(f"Error with sample {sample_id}: {e}")
            print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for sample_id, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  Sample {sample_id}: {status}")
