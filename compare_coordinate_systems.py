#!/usr/bin/env python3
"""
Compare the same prediction evaluated in different coordinate systems.
"""

import json
import numpy as np

def parse_kitti_label_line(line: str):
    """Parse KITTI label (camera coords)."""
    parts = line.strip().split()
    if len(parts) < 15:
        return None

    return {
        'type': parts[0],
        'location': [float(parts[11]), float(parts[12]), float(parts[13])],  # x, y, z (camera)
        'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],  # h, w, l
        'rotation_y': float(parts[14])
    }


def camera_to_lidar(x_cam, y_cam, z_cam):
    """
    Transform from camera to LiDAR coordinates (KITTI calibration).

    Camera: x=right, y=down, z=forward
    LiDAR: x=forward, y=left, z=up

    Rough approximation (exact values depend on calibration):
    x_lidar = z_cam
    y_lidar = -x_cam
    z_lidar = -y_cam
    """
    x_lidar = z_cam
    y_lidar = -x_cam
    z_lidar = -y_cam
    return x_lidar, y_lidar, z_lidar


sample = '000007'

# Load GT in camera coords
gt_file = f'/data2/yoshida/label_kitti/training/label_2/{sample}.txt'
with open(gt_file, 'r') as f:
    lines = f.readlines()

print("=" * 80)
print(f"COORDINATE SYSTEM COMPARISON - Sample {sample}")
print("=" * 80)

# Parse GT
gt_cars = []
for line in lines:
    obj = parse_kitti_label_line(line)
    if obj and obj['type'] == 'Car':
        gt_cars.append(obj)

if not gt_cars:
    print("No Car GT found")
    exit(1)

# Load predictions
with open('/data2/yoshida/kitti_100/kitti_no_spoofer_bin/kitti_predictions_20251108_112225.json', 'r') as f:
    preds = json.load(f)['results'][sample]

print(f"\nGT Cars: {len(gt_cars)}")
print(f"Predictions: {len(preds)}")

# Show first GT in both coordinate systems
gt = gt_cars[0]
x_cam, y_cam, z_cam = gt['location']
x_lid, y_lid, z_lid = camera_to_lidar(x_cam, y_cam, z_cam)

print(f"\nFirst GT Car:")
print(f"  Camera coords (x=right, y=down, z=forward):")
print(f"    x={x_cam:7.2f}, y={y_cam:7.2f}, z={z_cam:7.2f}")
print(f"  LiDAR coords  (x=forward, y=left, z=up):")
print(f"    x={x_lid:7.2f}, y={y_lid:7.2f}, z={z_lid:7.2f}")

# Find closest prediction in each coordinate system
print(f"\nPredictions:")
for i, pred in enumerate(preds[:5]):
    px, py, pz = pred['translation']
    score = pred['detection_score']

    # Distance if prediction is in camera coords
    dist_cam = np.sqrt((px - x_cam)**2 + (pz - z_cam)**2)  # BEV in camera (x-z)

    # Distance if prediction is in LiDAR coords
    dist_lid = np.sqrt((px - x_lid)**2 + (py - y_lid)**2)  # BEV in LiDAR (x-y)

    print(f"\n  Pred {i+1} (score={score:.3f}): x={px:7.2f}, y={py:7.2f}, z={pz:7.2f}")
    print(f"    If camera coords → BEV dist from GT: {dist_cam:6.2f}m")
    print(f"    If LiDAR coords  → BEV dist from GT: {dist_lid:6.2f}m")

    if i == 0:
        print(f"\n    Analysis:")
        if dist_lid < 2.0 and dist_cam > 10.0:
            print(f"    ✅ LIKELY LIDAR COORDS (small LiDAR dist, large camera dist)")
        elif dist_cam < 2.0 and dist_lid > 10.0:
            print(f"    ✅ LIKELY CAMERA COORDS (small camera dist, large LiDAR dist)")
        else:
            print(f"    ⚠️  UNCLEAR (both distances are similar)")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("If LiDAR distances are small → predictions are in LiDAR coords")
print("If Camera distances are small → predictions are in Camera coords")
print("=" * 80)
