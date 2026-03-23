"""
Calculate mAP using nuScenes-style metrics without coordinate transformation.

This script evaluates predictions in KITTI coordinate system using nuScenes
evaluation methodology:
- Center distance-based matching (instead of IoU@0.7)
- Multiple distance ranges
- nuScenes-style AP calculation
- No coordinate transformation (stays in KITTI/sensor coordinates)

COORDINATE SYSTEM NOTE:
- KITTI LiDAR: x=forward, y=left, z=up
- nuScenes: x=right, y=forward, z=up
- This script uses KITTI coordinates WITHOUT transformation
- BEV center distance (scalar) is invariant to coordinate rotation
- AP calculation is valid and correct
- TP metrics (trans_err, orient_err, etc.) should be interpreted in KITTI frame
"""

import argparse
import json
import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kitti_label_parser import KITTILabelParser
from evaluate_kitti_predictions import parse_predictions

try:
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.algo import accumulate, calc_ap
    from nuscenes.eval.detection.constants import TP_METRICS
    from nuscenes.eval.detection.data_classes import (
        DetectionBox,
        DetectionConfig,
        DetectionMetricDataList,
        DetectionMetrics,
    )
except ImportError as e:
    print(f"Error: nuScenes devkit not installed. Please install it:")
    print("  pip install nuscenes-devkit")
    sys.exit(1)


# nuScenes detection classes (we'll focus on 'car' for now)
NUSCENES_DETECTION_CLASSES = [
    'car',
    'truck',
    'bus',
    'trailer',
    'construction_vehicle',
    'pedestrian',
    'motorcycle',
    'bicycle',
    'traffic_cone',
    'barrier'
]

# Map KITTI classes to nuScenes classes
KITTI_TO_NUSCENES_CLASS = {
    'Car': 'car',
    'Pedestrian': 'pedestrian',
    'Cyclist': 'bicycle',
    'Van': 'car',
    'Truck': 'truck',
    'Person_sitting': 'pedestrian',
    'Tram': 'bus',
    'Misc': 'car',  # Fallback
    'DontCare': None,
}


def is_in_angle_range(x: float, y: float, center_angle: float, angle_width: float) -> bool:
    """
    Check if a point is within the specified angle range.

    Args:
        x: X coordinate (forward in KITTI lidar frame)
        y: Y coordinate (left in KITTI lidar frame)
        center_angle: Center angle in degrees (0°=forward, 90°=left, -90°=right)
        angle_width: Total width of angle range in degrees (e.g., 180 for ±90°)

    Returns:
        True if within range
    """
    angle = np.arctan2(y, x) * 180.0 / np.pi

    half_width = angle_width / 2.0
    min_angle = center_angle - half_width
    max_angle = center_angle + half_width

    def normalize(a):
        while a > 180:
            a -= 360
        while a < -180:
            a += 360
        return a

    angle = normalize(angle)
    min_angle = normalize(min_angle)
    max_angle = normalize(max_angle)

    if min_angle <= max_angle:
        return min_angle <= angle <= max_angle
    else:
        return angle >= min_angle or angle <= max_angle


def kitti_to_detection_box(bbox_3d: np.ndarray, score: float, class_name: str,
                           sample_token: str, velocity: Tuple[float, float] = (0.0, 0.0)) -> DetectionBox:
    """
    Convert KITTI-style bbox to nuScenes DetectionBox (without coordinate transformation).

    Args:
        bbox_3d: [x, y, z, w, l, h, ry] in KITTI lidar coordinates
        score: Detection confidence score
        class_name: nuScenes class name
        sample_token: Sample identifier
        velocity: (vx, vy) velocity in m/s

    Returns:
        DetectionBox in sensor coordinates
    """
    x, y, z, w, l, h, ry = bbox_3d

    # nuScenes uses [x, y, z] center, [w, l, h] size
    # KITTI also uses center coordinates, so no transformation needed
    translation = [float(x), float(y), float(z)]
    size = [float(w), float(l), float(h)]

    # Convert rotation angle to quaternion
    # KITTI uses rotation around Z-axis (yaw)
    # Quaternion: [w, x, y, z] = [cos(theta/2), 0, 0, sin(theta/2)]
    rotation = [
        np.cos(ry / 2),  # w
        0.0,              # x
        0.0,              # y
        np.sin(ry / 2)   # z
    ]

    return DetectionBox(
        sample_token=sample_token,
        translation=translation,
        size=size,
        rotation=rotation,
        velocity=velocity,
        detection_name=class_name,
        detection_score=score,
        attribute_name=''  # Not used for detection metrics
    )


def create_nuscenes_style_config() -> DetectionConfig:
    """
    Create nuScenes-style detection configuration using config_factory.

    This uses the same matching and AP calculation as nuScenes,
    but without requiring global coordinates.
    """
    from nuscenes.eval.common.config import config_factory

    # Use the official nuScenes detection config
    cfg = config_factory("detection_cvpr_2019")

    return cfg


def evaluate_nuscenes_style(
    pred_file: str,
    gt_label_dir: str,
    target_class: str = 'Car',
    score_threshold: float = 0.0,
    filter_kitti_range: bool = False,
    x_range: Tuple[float, float] = (0.0, 70.0),
    y_range: Tuple[float, float] = (-40.0, 40.0),
    center_angle: float = None,
    angle_width: float = None,
    verbose: bool = False
):
    """
    Evaluate predictions using nuScenes-style metrics (in KITTI coordinates).

    Args:
        pred_file: Path to prediction JSON file
        gt_label_dir: Directory containing KITTI ground truth labels
        target_class: KITTI class name to evaluate ('Car', 'Pedestrian', etc.)
        score_threshold: Minimum detection score to keep
        filter_kitti_range: If True, filter predictions to KITTI evaluation range
        x_range: X-axis range (forward direction) in meters
        y_range: Y-axis range (lateral direction) in meters
        verbose: Print detailed information
    """
    print("=" * 80)
    print("nuScenes-Style BEV mAP EVALUATION (KITTI Coordinates)")
    print("=" * 80)
    print(f"Target KITTI class: {target_class}")

    # Map KITTI class to nuScenes class
    if target_class not in KITTI_TO_NUSCENES_CLASS:
        print(f"Error: Unknown KITTI class '{target_class}'")
        return

    nuscenes_class = KITTI_TO_NUSCENES_CLASS[target_class]
    if nuscenes_class is None:
        print(f"Error: KITTI class '{target_class}' cannot be evaluated")
        return

    print(f"Mapped to nuScenes class: {nuscenes_class}")

    if score_threshold > 0:
        print(f"Score threshold: {score_threshold} (predictions below this will be filtered)")

    if filter_kitti_range:
        print(f"KITTI range filtering: ENABLED")
        print(f"  X range: [{x_range[0]:.1f}, {x_range[1]:.1f}] meters (forward)")
        print(f"  Y range: [{y_range[0]:.1f}, {y_range[1]:.1f}] meters (lateral)")
    else:
        print(f"KITTI range filtering: DISABLED (evaluating all predictions)")

    use_angle_filter = center_angle is not None and angle_width is not None
    if use_angle_filter:
        print(f"Angle filtering: ENABLED")
        print(f"  Center: {center_angle}°, Width: {angle_width}° "
              f"(range: {center_angle - angle_width/2:.1f}° to {center_angle + angle_width/2:.1f}°)")

    # Create detection config using official nuScenes config
    cfg = create_nuscenes_style_config()

    print(f"\nnuScenes-style configuration:")
    print(f"  Classes: {cfg.class_names}")
    print(f"  Distance function: {cfg.dist_fcn}")
    print(f"  Distance thresholds: {cfg.dist_ths}")
    print(f"  Min recall: {cfg.min_recall}")
    print(f"  Min precision: {cfg.min_precision}")

    # Load predictions
    print(f"\nLoading predictions: {pred_file}")
    predictions = parse_predictions(pred_file)
    print(f"  Loaded {len(predictions)} samples")

    # Convert predictions and GTs to nuScenes DetectionBox format
    pred_boxes_dict = {}
    gt_boxes_dict = {}

    total_preds = 0
    total_preds_before_filter = 0
    total_preds_after_score_filter = 0
    total_preds_after_range_filter = 0
    total_gts = 0

    for sample_idx, (sample_token, dets) in enumerate(predictions.items()):
        # Filter predictions by target class
        dets = [d for d in dets if d.class_name == target_class]
        total_preds_before_filter += len(dets)

        # Filter by score threshold
        if score_threshold > 0:
            dets = [d for d in dets if d.score >= score_threshold]
        total_preds_after_score_filter += len(dets)

        # Filter by KITTI evaluation range
        if filter_kitti_range:
            filtered_dets = []
            for det in dets:
                x, y = det.bbox.x, det.bbox.y
                if (x_range[0] <= x <= x_range[1] and
                    y_range[0] <= y <= y_range[1]):
                    filtered_dets.append(det)
            dets = filtered_dets
        total_preds_after_range_filter += len(dets)

        # Filter by angle range
        if use_angle_filter:
            dets = [d for d in dets
                    if is_in_angle_range(d.bbox.x, d.bbox.y, center_angle, angle_width)]

        # Convert to DetectionBox
        pred_boxes = []
        for det in dets:
            bbox_3d = np.array([
                det.bbox.x, det.bbox.y, det.bbox.z,  # x, y, z
                det.bbox.w, det.bbox.l, det.bbox.h,  # w, l, h
                det.bbox.ry  # ry
            ])
            box = kitti_to_detection_box(
                bbox_3d, det.score, nuscenes_class, sample_token
            )
            pred_boxes.append(box)

        pred_boxes_dict[sample_token] = pred_boxes
        total_preds += len(pred_boxes)

        # Load ground truth
        gt_file = os.path.join(gt_label_dir, f"{sample_token}.txt")

        if not os.path.exists(gt_file):
            if verbose:
                print(f"Warning: GT file not found for sample {sample_token}")
            gt_boxes_dict[sample_token] = []
            continue

        # Parse GT in lidar coordinates
        gts = KITTILabelParser.parse_ground_truth_file(gt_file, coordinate_system='lidar')

        # Filter GT by class
        gts = [g for g in gts if g.class_name == target_class]

        # Filter GT by angle range
        if use_angle_filter:
            gts = [g for g in gts
                   if is_in_angle_range(g.bbox.x, g.bbox.y, center_angle, angle_width)]

        # Convert to DetectionBox
        gt_boxes = []
        for gt in gts:
            bbox_3d = np.array([
                gt.bbox.x, gt.bbox.y, gt.bbox.z,  # x, y, z
                gt.bbox.w, gt.bbox.l, gt.bbox.h,  # w, l, h
                gt.bbox.ry  # ry
            ])
            box = kitti_to_detection_box(
                bbox_3d, 1.0, nuscenes_class, sample_token  # GT score is always 1.0
            )
            gt_boxes.append(box)

        gt_boxes_dict[sample_token] = gt_boxes
        total_gts += len(gt_boxes)

        # Progress
        if (sample_idx + 1) % 20 == 0:
            print(f"  Processed {sample_idx + 1}/{len(predictions)} samples...")

    # Print filtering statistics
    print(f"\nFiltering statistics:")
    print(f"  Initial predictions: {total_preds_before_filter}")

    if score_threshold > 0:
        print(f"  After score threshold (>= {score_threshold}): {total_preds_after_score_filter} "
              f"({total_preds_after_score_filter/total_preds_before_filter*100:.1f}%)")

    if filter_kitti_range:
        print(f"  After KITTI range filter: {total_preds_after_range_filter} "
              f"({total_preds_after_range_filter/total_preds_after_score_filter*100:.1f}% of scored)")
        excluded = total_preds_after_score_filter - total_preds_after_range_filter
        if excluded > 0:
            print(f"  Excluded {excluded} predictions outside KITTI range")

    if use_angle_filter:
        print(f"  After angle filter ({center_angle}° ±{angle_width/2:.0f}°): {total_preds} "
              f"({total_preds/max(total_preds_after_range_filter,1)*100:.1f}% of range-filtered)")

    print(f"\nFinal predictions for evaluation: {total_preds}")
    print(f"Total ground truth: {total_gts}")

    # Create EvalBoxes from dict
    pred_boxes = EvalBoxes()
    gt_boxes = EvalBoxes()

    pred_boxes.boxes = pred_boxes_dict
    gt_boxes.boxes = gt_boxes_dict

    # Accumulate metrics using nuScenes algorithm
    print(f"\nAccumulating metrics for class '{nuscenes_class}'...")

    metric_data_list = DetectionMetricDataList()

    # Collect all distance thresholds (including dist_th_tp for TP metrics)
    all_dist_ths = list(cfg.dist_ths)
    if cfg.dist_th_tp not in all_dist_ths:
        all_dist_ths.append(cfg.dist_th_tp)

    for dist_th in all_dist_ths:
        print(f"  Computing metrics for distance threshold: {dist_th}m")
        md = accumulate(
            gt_boxes,
            pred_boxes,
            nuscenes_class,
            cfg.dist_fcn_callable,  # Use distance function from config
            dist_th
        )
        metric_data_list.set(nuscenes_class, dist_th, md)

    # Calculate metrics
    metrics = DetectionMetrics(cfg)

    print("\nCalculating Average Precision...")

    # Only calculate AP for the target class (not all classes in config)
    aps = []
    for dist_th in cfg.dist_ths:
        metric_data = metric_data_list[(nuscenes_class, dist_th)]
        ap = calc_ap(metric_data, cfg.min_recall, cfg.min_precision)
        metrics.add_label_ap(nuscenes_class, dist_th, ap)
        aps.append(ap)

    # Calculate mean AP across distance thresholds for this class
    mean_ap = np.mean(aps) if aps else 0.0

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nClass: {target_class} → {nuscenes_class}")
    print(f"Matching method: Center distance (nuScenes-style)")
    print(f"\nAverage Precision by distance threshold:")
    print(f"{'Distance Threshold':<25} {'AP':<15} {'AP%':<15}")
    print("-" * 55)

    aps_dict = {}
    for i, dist_th in enumerate(cfg.dist_ths):
        ap = aps[i]
        aps_dict[dist_th] = ap
        print(f"{dist_th:<25.1f} {ap:<15.4f} {ap*100:<15.2f}")

    print(f"\n{'Mean AP:':<25} {mean_ap:<15.4f} {mean_ap*100:<15.2f}")

    # Print TP metrics if available
    print("\n" + "=" * 80)
    print(f"TRUE POSITIVE METRICS (at distance threshold = {cfg.dist_th_tp}m)")
    print("=" * 80)

    metric_data = metric_data_list[(nuscenes_class, cfg.dist_th_tp)]

    # Calculate TP metrics using nuScenes calc_tp function
    from nuscenes.eval.detection.algo import calc_tp

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

    # Save results to file
    output_file = pred_file.replace('.json', '_map_nuscenes_style.txt')
    with open(output_file, 'w') as f:
        f.write("nuScenes-Style BEV mAP Evaluation Results (KITTI Coordinates)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Prediction file: {pred_file}\n")
        f.write(f"GT directory: {gt_label_dir}\n")
        f.write(f"KITTI class: {target_class}\n")
        f.write(f"nuScenes class: {nuscenes_class}\n")
        f.write(f"Matching method: Center distance\n")
        f.write(f"Distance threshold for TP: {cfg.dist_th_tp}m\n")
        if score_threshold > 0:
            f.write(f"Score threshold: {score_threshold}\n")
        if filter_kitti_range:
            f.write(f"KITTI range filter: ENABLED\n")
            f.write(f"  X range: [{x_range[0]:.1f}, {x_range[1]:.1f}] meters\n")
            f.write(f"  Y range: [{y_range[0]:.1f}, {y_range[1]:.1f}] meters\n")
        if use_angle_filter:
            f.write(f"Angle filter: center={center_angle}°, width={angle_width}° "
                    f"(range: {center_angle - angle_width/2:.1f}° to {center_angle + angle_width/2:.1f}°)\n")
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
        description='Calculate mAP using nuScenes-style metrics (KITTI coordinates)'
    )
    parser.add_argument('pred_file', type=str,
                        help='Path to prediction JSON file')
    parser.add_argument('--gt-label-dir', type=str,
                        default='/data2/yoshida/label_kitti/training/label_2',
                        help='Directory containing ground truth labels')
    parser.add_argument('--class', dest='target_class', type=str, default='Car',
                        help='Target KITTI class to evaluate (Car, Pedestrian, etc.)')
    parser.add_argument('--score-threshold', type=float, default=0.0,
                        help='Minimum detection score to keep (default: 0.0, no filtering)')
    parser.add_argument('--filter-kitti-range', action='store_true',
                        help='Filter predictions to KITTI evaluation range (forward direction only)')
    parser.add_argument('--x-min', type=float, default=0.0,
                        help='Minimum X coordinate (forward) in meters (default: 0.0)')
    parser.add_argument('--x-max', type=float, default=70.0,
                        help='Maximum X coordinate (forward) in meters (default: 70.0)')
    parser.add_argument('--y-min', type=float, default=-40.0,
                        help='Minimum Y coordinate (lateral) in meters (default: -40.0)')
    parser.add_argument('--y-max', type=float, default=40.0,
                        help='Maximum Y coordinate (lateral) in meters (default: 40.0)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--center-angle', type=float, default=None,
                        help='Center angle in degrees for angle-based filtering '
                             '(KITTI lidar: 0°=forward/+X, 90°=left/+Y, -90°=right/-Y). '
                             'Requires --angle-width.')
    parser.add_argument('--angle-width', type=float, default=None,
                        help='Total width of angle range in degrees '
                             '(e.g., 180 for ±90° front half, 90 for ±45°). '
                             'Requires --center-angle.')

    args = parser.parse_args()

    evaluate_nuscenes_style(
        args.pred_file,
        args.gt_label_dir,
        args.target_class,
        args.score_threshold,
        args.filter_kitti_range,
        (args.x_min, args.x_max),
        (args.y_min, args.y_max),
        args.center_angle,
        args.angle_width,
        args.verbose
    )
