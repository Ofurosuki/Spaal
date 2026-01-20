"""
Calculate mAP using nuScenes-style metrics in KITTI CAMERA coordinate system.

This script evaluates predictions WITHOUT any coordinate transformation:
- GT: KITTI camera coordinates (directly from label files)
- Predictions: Assumed to be in camera coordinates
- Center distance matching (nuScenes-style)
- NO coordinate transformation to avoid parser bugs

COORDINATE SYSTEM:
- KITTI Camera: x=right, y=down, z=forward
- Both GT and predictions use this coordinate system
- BEV distance uses x and z (right and forward)
"""

import argparse
import json
import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from collections import defaultdict

try:
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from nuscenes.eval.detection.constants import TP_METRICS
    from nuscenes.eval.detection.data_classes import (
        DetectionBox,
        DetectionConfig,
        DetectionMetricDataList,
        DetectionMetrics,
    )
    from nuscenes.eval.common.config import config_factory
except ImportError as e:
    print(f"Error: nuScenes devkit not installed. Please install it:")
    print("  pip install nuscenes-devkit")
    sys.exit(1)


# Map nuScenes classes to KITTI classes
NUSCENES_TO_KITTI_CLASS = {
    'vehicle.car': 'Car',
    'vehicle.truck': 'Truck',
    'vehicle.bus': 'Tram',
    'vehicle.trailer': 'Misc',
    'vehicle.construction': 'Misc',
    'human.pedestrian': 'Pedestrian',
    'vehicle.motorcycle': 'Cyclist',
    'vehicle.bicycle': 'Cyclist',
    'movable_object.trafficcone': 'Misc',
    'movable_object.barrier': 'Misc',
}

KITTI_TO_NUSCENES_CLASS = {
    'Car': 'car',
    'Pedestrian': 'pedestrian',
    'Cyclist': 'bicycle',
    'Van': 'car',
    'Truck': 'truck',
    'Person_sitting': 'pedestrian',
    'Tram': 'bus',
    'Misc': 'car',
}


def parse_kitti_label_line(line: str) -> Dict:
    """
    Parse a single line from KITTI label file (camera coordinates).

    Format: type truncated occluded alpha bbox_2d dimensions location rotation_y [score]

    Returns dict with camera coordinates (NO transformation).
    """
    parts = line.strip().split()
    if len(parts) < 15:
        return None

    obj = {
        'type': parts[0],
        'truncated': float(parts[1]),
        'occluded': int(parts[2]),
        'alpha': float(parts[3]),
        'bbox': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
        'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],  # h, w, l
        'location': [float(parts[11]), float(parts[12]), float(parts[13])],  # x, y, z (camera)
        'rotation_y': float(parts[14])
    }

    return obj


def quaternion_to_yaw(quat: List[float]) -> float:
    """Convert quaternion [w, x, y, z] to yaw angle."""
    w, x, y, z = quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw


def camera_bbox_to_detection_box(
    x: float, y: float, z: float,  # Camera coordinates
    h: float, w: float, l: float,  # Dimensions
    ry: float,  # Rotation around Y axis (camera frame)
    score: float,
    class_name: str,
    sample_token: str
) -> DetectionBox:
    """
    Create DetectionBox in CAMERA coordinates (NO transformation).

    Args:
        x, y, z: Position in camera coordinates (x=right, y=down, z=forward)
        h, w, l: Height, width, length
        ry: Rotation around Y axis
        score: Detection score
        class_name: nuScenes class name
        sample_token: Sample identifier

    Returns:
        DetectionBox with camera coordinates
    """
    # Camera coordinates: x=right, y=down, z=forward
    translation = [float(x), float(y), float(z)]

    # Size: nuScenes uses [w, l, h], KITTI uses [h, w, l]
    size = [float(w), float(l), float(h)]

    # Rotation: Convert rotation_y to quaternion
    # In camera frame, rotation is around Y axis (down)
    # Quaternion for rotation around Y: [cos(θ/2), 0, sin(θ/2), 0]
    rotation = [
        np.cos(ry / 2),  # w
        0.0,              # x
        np.sin(ry / 2),   # y
        0.0               # z
    ]

    return DetectionBox(
        sample_token=sample_token,
        translation=translation,
        size=size,
        rotation=rotation,
        velocity=(0.0, 0.0),
        detection_name=class_name,
        detection_score=score,
        attribute_name=''
    )


def custom_camera_bev_distance(gt_box: DetectionBox, pred_box: DetectionBox) -> float:
    """
    Calculate BEV distance in CAMERA coordinates.

    In camera frame (x=right, y=down, z=forward):
    BEV is x-z plane (right-forward plane)

    Returns:
        L2 distance in BEV (x-z plane)
    """
    # Camera BEV uses x and z (indices 0 and 2)
    gt_bev = np.array([gt_box.translation[0], gt_box.translation[2]])
    pred_bev = np.array([pred_box.translation[0], pred_box.translation[2]])

    return np.linalg.norm(pred_bev - gt_bev)


def evaluate_nuscenes_camera_coords(
    pred_file: str,
    gt_label_dir: str,
    target_class: str = 'Car',
    score_threshold: float = 0.0,
    filter_range: bool = False,
    x_range: Tuple[float, float] = (-40.0, 40.0),  # Right/left in camera
    z_range: Tuple[float, float] = (0.0, 70.0),   # Forward in camera
    verbose: bool = False
):
    """
    Evaluate using nuScenes metrics in CAMERA coordinates (NO transformation).
    """
    print("=" * 80)
    print("nuScenes-Style mAP EVALUATION (CAMERA COORDINATES - NO TRANSFORMATION)")
    print("=" * 80)
    print(f"Target KITTI class: {target_class}")
    print(f"Coordinate system: KITTI Camera (x=right, y=down, z=forward)")
    print(f"BEV plane: x-z (right-forward)")

    # Map to nuScenes class
    if target_class not in KITTI_TO_NUSCENES_CLASS:
        print(f"Error: Unknown KITTI class '{target_class}'")
        return

    nuscenes_class = KITTI_TO_NUSCENES_CLASS[target_class]
    print(f"Mapped to nuScenes class: {nuscenes_class}")

    if score_threshold > 0:
        print(f"Score threshold: {score_threshold}")

    if filter_range:
        print(f"Range filtering: ENABLED")
        print(f"  X range (right): [{x_range[0]:.1f}, {x_range[1]:.1f}] meters")
        print(f"  Z range (forward): [{z_range[0]:.1f}, {z_range[1]:.1f}] meters")

    # Load predictions
    print(f"\nLoading predictions: {pred_file}")
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    predictions = pred_data['results']
    print(f"  Loaded {len(predictions)} samples")

    # Convert predictions and GTs to DetectionBox format (camera coords)
    pred_boxes_dict = {}
    gt_boxes_dict = {}

    total_preds = 0
    total_preds_before_filter = 0
    total_preds_after_score_filter = 0
    total_gts = 0

    for sample_idx, (sample_token, pred_list) in enumerate(predictions.items()):
        # Process predictions
        sample_preds = []

        for pred in pred_list:
            # Get nuScenes class name
            pred_class = pred['detection_name']

            # Map to KITTI class
            if pred_class not in NUSCENES_TO_KITTI_CLASS:
                continue

            kitti_class = NUSCENES_TO_KITTI_CLASS[pred_class]

            # Filter by target class
            if kitti_class != target_class:
                continue

            total_preds_before_filter += 1

            # Get prediction data (assuming camera coordinates)
            x, y, z = pred['translation']
            w, l, h = pred['size']
            score = pred['detection_score']

            # Filter by score
            if score < score_threshold:
                continue

            total_preds_after_score_filter += 1

            # Filter by range
            if filter_range:
                if not (x_range[0] <= x <= x_range[1] and z_range[0] <= z <= z_range[1]):
                    continue

            # Convert quaternion to rotation_y
            quat = pred['rotation']
            ry = quaternion_to_yaw(quat)

            # Create DetectionBox (camera coordinates)
            box = camera_bbox_to_detection_box(
                x, y, z, h, w, l, ry, score, nuscenes_class, sample_token
            )
            sample_preds.append(box)

        pred_boxes_dict[sample_token] = sample_preds
        total_preds += len(sample_preds)

        # Load ground truth (camera coordinates, NO transformation)
        gt_file = os.path.join(gt_label_dir, f"{sample_token}.txt")

        if not os.path.exists(gt_file):
            if verbose:
                print(f"Warning: GT file not found for sample {sample_token}")
            gt_boxes_dict[sample_token] = []
            continue

        # Parse GT labels (stay in camera coordinates)
        sample_gts = []
        with open(gt_file, 'r') as f:
            for line in f:
                obj = parse_kitti_label_line(line)
                if obj is None or obj['type'] == 'DontCare':
                    continue

                # Filter by target class
                if obj['type'] != target_class:
                    continue

                # Extract camera coordinates (NO transformation)
                x, y, z = obj['location']  # Camera coordinates
                h, w, l = obj['dimensions']
                ry = obj['rotation_y']

                # Create DetectionBox (camera coordinates)
                box = camera_bbox_to_detection_box(
                    x, y, z, h, w, l, ry, 1.0, nuscenes_class, sample_token
                )
                sample_gts.append(box)

        gt_boxes_dict[sample_token] = sample_gts
        total_gts += len(sample_gts)

        # Progress
        if (sample_idx + 1) % 20 == 0:
            print(f"  Processed {sample_idx + 1}/{len(predictions)} samples...")

    # Print filtering statistics
    print(f"\nFiltering statistics:")
    print(f"  Initial predictions: {total_preds_before_filter}")
    if score_threshold > 0:
        print(f"  After score filter: {total_preds_after_score_filter}")
    if filter_range:
        excluded = total_preds_after_score_filter - total_preds
        if excluded > 0:
            print(f"  After range filter: {total_preds}")
            print(f"  Excluded {excluded} predictions outside range")
    print(f"\nFinal predictions for evaluation: {total_preds}")
    print(f"Total ground truth: {total_gts}")

    # Create EvalBoxes
    pred_boxes = EvalBoxes()
    gt_boxes = EvalBoxes()
    pred_boxes.boxes = pred_boxes_dict
    gt_boxes.boxes = gt_boxes_dict

    # Create nuScenes config
    cfg = config_factory("detection_cvpr_2019")

    print(f"\nnuScenes configuration:")
    print(f"  Distance thresholds: {cfg.dist_ths}")
    print(f"  Distance function: {cfg.dist_fcn} (modified for camera BEV)")
    print(f"  Min recall: {cfg.min_recall}")
    print(f"  Min precision: {cfg.min_precision}")

    # Accumulate metrics using CAMERA BEV distance
    print(f"\nAccumulating metrics for class '{nuscenes_class}'...")

    metric_data_list = DetectionMetricDataList()

    all_dist_ths = list(cfg.dist_ths)
    if cfg.dist_th_tp not in all_dist_ths:
        all_dist_ths.append(cfg.dist_th_tp)

    for dist_th in all_dist_ths:
        print(f"  Computing metrics for distance threshold: {dist_th}m")
        md = accumulate(
            gt_boxes,
            pred_boxes,
            nuscenes_class,
            custom_camera_bev_distance,  # Use camera BEV distance!
            dist_th
        )
        metric_data_list.set(nuscenes_class, dist_th, md)

    # Calculate metrics
    print("\nCalculating Average Precision...")

    metrics = DetectionMetrics(cfg)
    aps = []
    for dist_th in cfg.dist_ths:
        metric_data = metric_data_list[(nuscenes_class, dist_th)]
        ap = calc_ap(metric_data, cfg.min_recall, cfg.min_precision)
        metrics.add_label_ap(nuscenes_class, dist_th, ap)
        aps.append(ap)

    mean_ap = np.mean(aps) if aps else 0.0

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nClass: {target_class} → {nuscenes_class}")
    print(f"Coordinate system: CAMERA (x=right, y=down, z=forward)")
    print(f"BEV matching: x-z plane (right-forward)")
    print(f"\nAverage Precision by distance threshold:")
    print(f"{'Distance Threshold':<25} {'AP':<15} {'AP%':<15}")
    print("-" * 55)

    aps_dict = {}
    for i, dist_th in enumerate(cfg.dist_ths):
        ap = aps[i]
        aps_dict[dist_th] = ap
        print(f"{dist_th:<25.1f} {ap:<15.4f} {ap*100:<15.2f}")

    print(f"\n{'Mean AP:':<25} {mean_ap:<15.4f} {mean_ap*100:<15.2f}")

    # Print TP metrics
    print("\n" + "=" * 80)
    print(f"TRUE POSITIVE METRICS (at distance threshold = {cfg.dist_th_tp}m)")
    print("=" * 80)

    metric_data = metric_data_list[(nuscenes_class, cfg.dist_th_tp)]

    tp_metrics_summary = {}
    for metric_name in TP_METRICS:
        try:
            tp = calc_tp(metric_data, cfg.min_recall, metric_name)
            if not np.isnan(tp):
                tp_metrics_summary[metric_name] = tp
                print(f"{metric_name:<20} {tp:.4f}")
            else:
                print(f"{metric_name:<20} N/A")
        except Exception as e:
            if verbose:
                print(f"{metric_name:<20} Error: {e}")
            else:
                print(f"{metric_name:<20} N/A")

    # Save results
    output_file = pred_file.replace('.json', '_map_nuscenes_camera.txt')
    with open(output_file, 'w') as f:
        f.write("nuScenes-Style BEV mAP Evaluation (CAMERA Coordinates - NO Transformation)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Prediction file: {pred_file}\n")
        f.write(f"GT directory: {gt_label_dir}\n")
        f.write(f"KITTI class: {target_class}\n")
        f.write(f"nuScenes class: {nuscenes_class}\n")
        f.write(f"Coordinate system: CAMERA (x=right, y=down, z=forward)\n")
        f.write(f"BEV matching: x-z plane (right-forward)\n")
        f.write(f"Distance threshold for TP: {cfg.dist_th_tp}m\n")
        if score_threshold > 0:
            f.write(f"Score threshold: {score_threshold}\n")
        if filter_range:
            f.write(f"Range filter: ENABLED\n")
            f.write(f"  X range: [{x_range[0]:.1f}, {x_range[1]:.1f}] meters\n")
            f.write(f"  Z range: [{z_range[0]:.1f}, {z_range[1]:.1f}] meters\n")
        f.write("\n")

        f.write(f"Total predictions: {total_preds}\n")
        f.write(f"Total ground truth: {total_gts}\n\n")

        f.write("Average Precision by distance threshold:\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Distance Threshold':<25} {'AP':<15} {'AP%':<15}\n")
        f.write("-" * 55 + "\n")

        for dist_th in cfg.dist_ths:
            ap = aps_dict[dist_th]
            f.write(f"{dist_th:<25.1f} {ap:<15.4f} {ap*100:<15.2f}\n")

        f.write(f"\n{'Mean AP:':<25} {mean_ap:<15.4f} {mean_ap*100:<15.2f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"True Positive Metrics (at distance threshold = {cfg.dist_th_tp}m):\n")
        f.write("=" * 80 + "\n")

        for metric_name, value in tp_metrics_summary.items():
            f.write(f"{metric_name:<20} {value:.4f}\n")

    print(f"\nResults saved to: {output_file}")
    print("=" * 80)

    return {
        'mean_ap': mean_ap,
        'ap_by_distance': aps_dict,
        'tp_metrics': tp_metrics_summary
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate mAP using nuScenes metrics in CAMERA coordinates (NO transformation)'
    )
    parser.add_argument('pred_file', type=str,
                        help='Path to prediction JSON file')
    parser.add_argument('--gt-label-dir', type=str,
                        default='/data2/yoshida/label_kitti/training/label_2',
                        help='Directory containing ground truth labels')
    parser.add_argument('--class', dest='target_class', type=str, default='Car',
                        help='Target KITTI class to evaluate (Car, Pedestrian, etc.)')
    parser.add_argument('--score-threshold', type=float, default=0.0,
                        help='Minimum detection score to keep (default: 0.0)')
    parser.add_argument('--filter-range', action='store_true',
                        help='Filter predictions to evaluation range')
    parser.add_argument('--x-min', type=float, default=-40.0,
                        help='Minimum X coordinate (right) in meters (default: -40.0)')
    parser.add_argument('--x-max', type=float, default=40.0,
                        help='Maximum X coordinate (right) in meters (default: 40.0)')
    parser.add_argument('--z-min', type=float, default=0.0,
                        help='Minimum Z coordinate (forward) in meters (default: 0.0)')
    parser.add_argument('--z-max', type=float, default=70.0,
                        help='Maximum Z coordinate (forward) in meters (default: 70.0)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')

    args = parser.parse_args()

    evaluate_nuscenes_camera_coords(
        args.pred_file,
        args.gt_label_dir,
        args.target_class,
        args.score_threshold,
        args.filter_range,
        (args.x_min, args.x_max),
        (args.z_min, args.z_max),
        args.verbose
    )
