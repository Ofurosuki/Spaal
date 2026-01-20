#!/usr/bin/env python3
"""
Test coordinate system impact on nuScenes evaluation.
"""

import numpy as np
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.utils import center_distance

def create_box(x, y, z, w, l, h, yaw, sample_token="test"):
    """Create a DetectionBox object."""
    translation = [x, y, z]
    size = [w, l, h]

    # Quaternion from yaw
    rotation = [
        np.cos(yaw / 2),  # w
        0.0,              # x
        0.0,              # y
        np.sin(yaw / 2)   # z
    ]

    return DetectionBox(
        sample_token=sample_token,
        translation=translation,
        size=size,
        rotation=rotation,
        detection_name="car",
        detection_score=1.0,
        velocity=(0.0, 0.0),
        attribute_name=""
    )

print("=" * 80)
print("COORDINATE SYSTEM TEST")
print("=" * 80)

# Test case 1: Car at origin, another car ahead
print("\nTest 1: Two cars - one at origin, one 10m ahead")
print("-" * 80)

# KITTI coordinate system (current implementation)
print("\nKITTI coordinates (x=forward, y=left, z=up):")
gt_kitti = create_box(x=0, y=0, z=0, w=4, l=2, h=1.6, yaw=0)
pred_kitti = create_box(x=10, y=0, z=0, w=4, l=2, h=1.6, yaw=0)
dist_kitti = center_distance(gt_kitti, pred_kitti)
print(f"  GT: x=0 (forward), y=0 (left)")
print(f"  Pred: x=10 (forward), y=0 (left)")
print(f"  Center distance (BEV): {dist_kitti:.2f}m")
print(f"  Interpretation: Car is 10m ahead")

# If we were to use nuScenes coordinates (x=right, y=forward)
print("\nnuScenes coordinates (x=right, y=forward, z=up):")
gt_nus = create_box(x=0, y=0, z=0, w=4, l=2, h=1.6, yaw=0)
pred_nus = create_box(x=0, y=10, z=0, w=4, l=2, h=1.6, yaw=0)
dist_nus = center_distance(gt_nus, pred_nus)
print(f"  GT: x=0 (right), y=0 (forward)")
print(f"  Pred: x=0 (right), y=10 (forward)")
print(f"  Center distance (BEV): {dist_nus:.2f}m")
print(f"  Interpretation: Car is 10m ahead")

# Test case 2: Car to the left
print("\n" + "=" * 80)
print("Test 2: Car 5m to the left")
print("-" * 80)

print("\nKITTI coordinates:")
pred_kitti_left = create_box(x=0, y=5, z=0, w=4, l=2, h=1.6, yaw=0)
dist_kitti_left = center_distance(gt_kitti, pred_kitti_left)
print(f"  GT: x=0, y=0")
print(f"  Pred: x=0 (no forward movement), y=5 (5m left)")
print(f"  Center distance (BEV): {dist_kitti_left:.2f}m")
print(f"  Interpretation: Car is 5m to the left")

print("\nnuScenes coordinates (if we transformed):")
pred_nus_left = create_box(x=-5, y=0, z=0, w=4, l=2, h=1.6, yaw=0)
dist_nus_left = center_distance(gt_nus, pred_nus_left)
print(f"  GT: x=0, y=0")
print(f"  Pred: x=-5 (5m to left), y=0 (no forward movement)")
print(f"  Center distance (BEV): {dist_nus_left:.2f}m")
print(f"  Interpretation: Car is 5m to the left")

# Test case 3: Diagonal movement
print("\n" + "=" * 80)
print("Test 3: Car 10m ahead and 5m to the left")
print("-" * 80)

print("\nKITTI coordinates (current implementation):")
pred_kitti_diag = create_box(x=10, y=5, z=0, w=4, l=2, h=1.6, yaw=0)
dist_kitti_diag = center_distance(gt_kitti, pred_kitti_diag)
print(f"  GT: x=0, y=0")
print(f"  Pred: x=10 (forward), y=5 (left)")
print(f"  Center distance (BEV): {dist_kitti_diag:.2f}m")
print(f"  Calculation: sqrt(10^2 + 5^2) = sqrt(125) = 11.18m")

print("\nnuScenes coordinates (if we transformed):")
# Transform: KITTI (x, y) -> nuScenes (y, -x)
# x_kitti=10, y_kitti=5 -> x_nus=5, y_nus=10
pred_nus_diag = create_box(x=5, y=10, z=0, w=4, l=2, h=1.6, yaw=0)
dist_nus_diag = center_distance(gt_nus, pred_nus_diag)
print(f"  GT: x=0, y=0")
print(f"  Pred: x=5 (right), y=10 (forward)")
print(f"  Center distance (BEV): {dist_nus_diag:.2f}m")
print(f"  Calculation: sqrt(5^2 + 10^2) = sqrt(125) = 11.18m")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("\n✅ Distance itself is INVARIANT to coordinate system rotation")
print("   (as long as we're in BEV xy-plane)")
print("\n⚠️  However, the MEANING of distance components changes:")
print("   - KITTI: distance includes (forward, left)")
print("   - nuScenes: distance includes (right, forward)")
print("\n⚠️  This affects:")
print("   1. Directional error metrics (trans_err components)")
print("   2. Orientation calculations")
print("   3. Velocity direction (if used)")
print("\n📌 Current implementation: Uses KITTI coordinates WITHOUT transformation")
print("   - Center distance (scalar) is correct")
print("   - But directional metrics may be misleading")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("\nOption 1: Keep KITTI coordinates (current)")
print("  Pros: No transformation needed, simpler")
print("  Cons: TP metrics (trans_err, etc.) in KITTI frame, not nuScenes")
print("\nOption 2: Transform to nuScenes coordinates")
print("  Pros: TP metrics meaningful in nuScenes context")
print("  Cons: Requires coordinate transformation")
print("\n→ For KITTI dataset evaluation, Option 1 (current) is ACCEPTABLE")
print("  because we only care about scalar distance for AP calculation.")
print("  TP metrics should be interpreted in KITTI coordinate frame.")
print("=" * 80)
