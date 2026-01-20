"""
Calculate mAP using official nuScenes evaluation metrics.

This script uses the nuScenes devkit to compute accurate mAP scores
following the official evaluation protocol. It takes the same prediction
JSON format as the KITTI evaluation script but evaluates using nuScenes metrics.
"""
import argparse
import json
import os
import sys
import numpy as np
import tempfile

# Import nuScenes evaluation modules
try:
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.loaders import (
        add_center_dist,
        filter_eval_boxes,
        load_gt_of_sample_tokens,
        load_prediction_of_sample_tokens,
    )
    from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from nuscenes.eval.detection.constants import TP_METRICS
    from nuscenes.eval.detection.data_classes import (
        DetectionBox,
        DetectionConfig,
        DetectionMetricDataList,
        DetectionMetrics,
    )
    from pyquaternion import Quaternion
except ImportError as e:
    print(f"Error: nuScenes devkit not installed. Please install it first:")
    print("  pip install nuscenes-devkit")
    sys.exit(1)


# Mapping from MMDetection3D class names to nuScenes class names
MMDET_TO_NUSCENES_CLASS = {
    'vehicle.car': 'car',
    'vehicle.truck': 'truck',
    'vehicle.bus': 'bus',
    'vehicle.trailer': 'trailer',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.construction': 'construction_vehicle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.barrier': 'barrier',
    # Short names (already in nuScenes format)
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'trailer': 'trailer',
    'construction_vehicle': 'construction_vehicle',
    'pedestrian': 'pedestrian',
    'motorcycle': 'motorcycle',
    'bicycle': 'bicycle',
    'traffic_cone': 'traffic_cone',
    'barrier': 'barrier',
    # KITTI-style names
    'Car': 'car',
    'Pedestrian': 'pedestrian',
    'Cyclist': 'bicycle',
}


def transform_sensor_to_global(translation, rotation, sample_token, nusc):
    """
    Transform box from sensor coordinates to global coordinates.

    Args:
        translation: [x, y, z] in sensor coordinates
        rotation: [w, x, y, z] quaternion in sensor coordinates
        sample_token: Sample token
        nusc: NuScenes instance

    Returns:
        Tuple of (global_translation, global_rotation)
    """
    # Get sensor and ego pose information
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    sd_record = nusc.get('sample_data', lidar_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Create quaternions
    box_quat = Quaternion(rotation)
    cs_quat = Quaternion(cs_record['rotation'])
    ego_quat = Quaternion(ego_pose['rotation'])

    # Transform translation: sensor -> ego -> global
    # sensor -> ego
    translation_ego = cs_quat.rotate(np.array(translation)) + np.array(cs_record['translation'])

    # ego -> global
    translation_global = ego_quat.rotate(translation_ego) + np.array(ego_pose['translation'])

    # Transform rotation: sensor -> ego -> global
    rotation_global = ego_quat * cs_quat * box_quat

    return translation_global.tolist(), [rotation_global.w, rotation_global.x, rotation_global.y, rotation_global.z]


def calculate_angle_in_sensor(translation):
    """
    Calculate angle in sensor coordinates (BEV).

    Args:
        translation: [x, y, z] in sensor coordinates

    Returns:
        Angle in degrees (0° = forward/X-axis, 90° = left/Y-axis)
    """
    x, y, _ = translation
    angle = np.arctan2(y, x) * 180.0 / np.pi
    return angle


def is_in_angle_range(translation, center_angle, angle_width):
    """
    Check if a box is within the specified angle range in sensor coordinates.

    Args:
        translation: [x, y, z] in sensor coordinates
        center_angle: Center angle in degrees
        angle_width: Total width of the angle range in degrees

    Returns:
        True if within range, False otherwise
    """
    if center_angle is None or angle_width is None:
        return True  # No filtering

    angle = calculate_angle_in_sensor(translation)

    # Calculate angle range
    half_width = angle_width / 2.0
    min_angle = center_angle - half_width
    max_angle = center_angle + half_width

    # Normalize angles to [-180, 180]
    def normalize_angle(a):
        while a > 180:
            a -= 360
        while a < -180:
            a += 360
        return a

    angle = normalize_angle(angle)
    min_angle = normalize_angle(min_angle)
    max_angle = normalize_angle(max_angle)

    # Check if angle is in range (handle wrap-around)
    if min_angle <= max_angle:
        return min_angle <= angle <= max_angle
    else:
        # Range wraps around ±180°
        return angle >= min_angle or angle <= max_angle


def filter_boxes_by_angle(data, nusc, sample_tokens, center_angle, angle_width):
    """
    Filter predictions by angle range in sensor coordinates (before coordinate transformation).

    Args:
        data: Prediction data dict
        nusc: NuScenes instance
        sample_tokens: List of sample tokens
        center_angle: Center angle in degrees
        angle_width: Total width in degrees

    Returns:
        Filtered data dict, number of boxes before, number of boxes after
    """
    if center_angle is None or angle_width is None:
        # Count total boxes
        total = sum(len(preds) for preds in data.get('results', {}).values())
        return data, total, total

    total_before = 0
    total_after = 0

    if 'results' in data:
        for sample_token in sample_tokens:
            if sample_token not in data['results']:
                continue

            predictions = data['results'][sample_token]
            filtered_predictions = []

            for pred in predictions:
                total_before += 1

                # Check if in angle range (using sensor coordinates before transformation)
                sensor_translation = pred['translation']

                if is_in_angle_range(sensor_translation, center_angle, angle_width):
                    filtered_predictions.append(pred)
                    total_after += 1

            data['results'][sample_token] = filtered_predictions

    return data, total_before, total_after


def convert_prediction_json(json_path: str, class_mapping: dict, nusc, sample_tokens: list,
                           center_angle=None, angle_width=None) -> str:
    """
    Convert MMDetection3D class names to nuScenes class names in prediction JSON.
    Also transforms predictions from sensor coordinates to global coordinates.
    Optionally filters by angle range in sensor coordinates.
    Creates a temporary converted JSON file.

    Args:
        json_path: Path to original predictions JSON
        class_mapping: Dictionary mapping MMDetection3D names to nuScenes names
        nusc: NuScenes instance for coordinate transformation
        sample_tokens: List of sample tokens to process
        center_angle: Center angle in degrees (sensor coordinates)
        angle_width: Total width of angle range in degrees

    Returns:
        Path to converted JSON file
    """
    # Load original predictions
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Filter by angle range (in sensor coordinates, before transformation)
    data, total_before_angle, total_after_angle = filter_boxes_by_angle(
        data, nusc, sample_tokens, center_angle, angle_width
    )

    if center_angle is not None and angle_width is not None:
        print(f"Angle filtering: {total_before_angle} boxes -> {total_after_angle} kept "
              f"(center={center_angle}°, width={angle_width}°)")

    unknown_classes = set()
    converted_count = 0
    skipped_count = 0
    transformed_count = 0

    # Convert class names and transform coordinates in results
    if 'results' in data:
        for sample_token in sample_tokens:
            if sample_token not in data['results']:
                continue

            predictions = data['results'][sample_token]
            converted_predictions = []

            for pred in predictions:
                original_name = pred.get('detection_name', '')

                if original_name in class_mapping:
                    # Convert class name
                    pred['detection_name'] = class_mapping[original_name]

                    # Transform from sensor to global coordinates
                    sensor_translation = pred['translation']
                    sensor_rotation = pred['rotation']

                    global_translation, global_rotation = transform_sensor_to_global(
                        sensor_translation,
                        sensor_rotation,
                        sample_token,
                        nusc
                    )

                    pred['translation'] = global_translation
                    pred['rotation'] = global_rotation

                    converted_predictions.append(pred)
                    converted_count += 1
                    transformed_count += 1
                else:
                    unknown_classes.add(original_name)
                    skipped_count += 1

            # Update predictions for this sample
            data['results'][sample_token] = converted_predictions

    if unknown_classes:
        print(f"Warning: Unknown classes found and skipped: {unknown_classes}")

    print(f"Converted {converted_count + skipped_count} predictions -> {converted_count} kept ({skipped_count} skipped)")
    print(f"Transformed {transformed_count} boxes from sensor to global coordinates")

    # Save to temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.json', prefix='nuscenes_pred_')
    with os.fdopen(temp_fd, 'w') as f:
        json.dump(data, f)

    return temp_path


def filter_gt_boxes_by_angle(gt_boxes: EvalBoxes, nusc, center_angle, angle_width) -> EvalBoxes:
    """
    Filter GT boxes by angle range in sensor coordinates.

    Args:
        gt_boxes: Ground truth boxes in global coordinates
        nusc: NuScenes instance
        center_angle: Center angle in degrees (sensor coordinates)
        angle_width: Total width in degrees

    Returns:
        Filtered EvalBoxes
    """
    if center_angle is None or angle_width is None:
        return gt_boxes

    filtered_boxes = EvalBoxes()
    total_before = 0
    total_after = 0

    for sample_token in gt_boxes.sample_tokens:
        # Get sensor info for this sample
        sample = nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        sd_record = nusc.get('sample_data', lidar_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', sd_record['ego_pose_token'])

        filtered_sample_boxes = []

        for box in gt_boxes.boxes[sample_token]:
            total_before += 1

            # Transform from global to sensor coordinates
            # global -> ego
            translation_ego = np.array(box.translation) - np.array(ego_pose['translation'])
            translation_ego = Quaternion(ego_pose['rotation']).inverse.rotate(translation_ego)

            # ego -> sensor
            translation_sensor = translation_ego - np.array(cs_record['translation'])
            translation_sensor = Quaternion(cs_record['rotation']).inverse.rotate(translation_sensor)

            # Check if in angle range
            if is_in_angle_range(translation_sensor, center_angle, angle_width):
                filtered_sample_boxes.append(box)
                total_after += 1

        if filtered_sample_boxes:
            filtered_boxes.add_boxes(sample_token, filtered_sample_boxes)

    print(f"GT angle filtering: {total_before} boxes -> {total_after} kept "
          f"(center={center_angle}°, width={angle_width}°)")

    return filtered_boxes


def load_sample_tokens(tokens_file: str) -> list:
    """Load sample tokens from JSON file."""
    with open(tokens_file, 'r') as f:
        tokens = json.load(f)
    print(f"Loaded {len(tokens)} sample tokens from {tokens_file}")
    return tokens


def evaluate_map_nuscenes_official(
    pred_file: str,
    sample_tokens_file: str,
    dataroot: str,
    version: str = "v1.0-mini",
    config_name: str = "detection_cvpr_2019",
    output_dir: str = None,
    center_angle: float = None,
    angle_width: float = None,
    verbose: bool = False
):
    """
    Evaluate mAP using official nuScenes evaluation

    Args:
        pred_file: Path to prediction JSON file
        sample_tokens_file: Path to sample tokens JSON file
        dataroot: nuScenes data root directory
        version: nuScenes version
        config_name: Evaluation config name
        output_dir: Output directory for results
        center_angle: Center angle for filtering (sensor coords)
        angle_width: Width of angle range for filtering
        verbose: Verbose output
    """
    # Create output directory
    if output_dir is None:
        output_dir = os.path.dirname(pred_file)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("nuScenes Official mAP Calculation")
    print("=" * 80)
    print(f"Predictions: {pred_file}")
    print(f"Sample Tokens: {sample_tokens_file}")
    print(f"Data Root: {dataroot}")
    print(f"Version: {version}")
    print(f"Config: {config_name}")
    if center_angle is not None and angle_width is not None:
        print(f"Angle Filter: center={center_angle}°, width={angle_width}° "
              f"(range: {center_angle - angle_width/2:.1f}° to {center_angle + angle_width/2:.1f}°)")
    print()

    # Load nuScenes
    print("Loading nuScenes...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
    print("nuScenes loaded.\n")

    # Load config
    cfg = config_factory(config_name)
    print(f"Evaluation config: {config_name}")
    print(f"  Classes: {cfg.class_names}")
    print(f"  Distance thresholds: {cfg.dist_ths}")
    print(f"  Distance function: {cfg.dist_fcn}")
    print(f"  Max boxes per sample: {cfg.max_boxes_per_sample}")
    print()

    # Load sample tokens
    print("Loading sample tokens...")
    sample_tokens = load_sample_tokens(sample_tokens_file)
    print()

    # Convert class names and transform coordinates in prediction JSON before loading
    print("Converting predictions from MMDetection3D sensor coordinates to nuScenes global coordinates...")
    converted_json_path = convert_prediction_json(
        pred_file,
        MMDET_TO_NUSCENES_CLASS,
        nusc,
        sample_tokens,
        center_angle=center_angle,
        angle_width=angle_width
    )
    print(f"Converted JSON saved to: {converted_json_path}")
    print()

    # Load converted predictions
    print("Loading converted predictions...")
    try:
        pred_boxes, meta = load_prediction_of_sample_tokens(
            converted_json_path,
            cfg.max_boxes_per_sample,
            DetectionBox,
            sample_tokens=sample_tokens,
            verbose=verbose
        )
        print(f"Loaded predictions for {len(pred_boxes.sample_tokens)} samples")
        print()
    finally:
        # Clean up temporary file
        if os.path.exists(converted_json_path):
            os.remove(converted_json_path)

    # Load ground truth
    print("Loading ground truth...")
    gt_boxes = load_gt_of_sample_tokens(
        nusc,
        sample_tokens,
        DetectionBox,
        verbose=verbose
    )
    print(f"Loaded GT for {len(gt_boxes.sample_tokens)} samples\n")

    # Verify sample tokens match
    assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
        "Sample tokens don't match between predictions and GT"

    # Add center distances
    print("Adding center distances...")
    pred_boxes = add_center_dist(nusc, pred_boxes)
    gt_boxes = add_center_dist(nusc, gt_boxes)
    print()

    # Filter boxes by distance range
    print("Filtering predictions by distance range...")
    pred_boxes = filter_eval_boxes(nusc, pred_boxes, cfg.class_range, verbose=verbose)
    print()

    print("Filtering ground truth by distance range...")
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, verbose=verbose)
    print()

    # Filter boxes by angle range (if specified)
    if center_angle is not None and angle_width is not None:
        print("Filtering ground truth by angle range...")
        gt_boxes = filter_gt_boxes_by_angle(gt_boxes, nusc, center_angle, angle_width)
        print()

    # Accumulate metric data
    print("Accumulating metric data...")
    metric_data_list = DetectionMetricDataList()

    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            if verbose:
                print(f"  {class_name} @ {dist_th}m")
            md = accumulate(gt_boxes, pred_boxes, class_name, cfg.dist_fcn_callable, dist_th)
            metric_data_list.set(class_name, dist_th, md)
    print()

    # Calculate metrics
    print("Calculating metrics...")
    metrics = DetectionMetrics(cfg)

    print("-" * 80)
    print(f"{'Class':<30s} {'AP@0.5m':<10s} {'AP@1.0m':<10s} {'AP@2.0m':<10s} {'AP@4.0m':<10s} {'mAP':<10s}")
    print("-" * 80)

    for class_name in cfg.class_names:
        # Compute APs for each distance threshold
        aps = []
        ap_strs = []
        for dist_th in cfg.dist_ths:
            metric_data = metric_data_list[(class_name, dist_th)]
            ap = calc_ap(metric_data, cfg.min_recall, cfg.min_precision)
            metrics.add_label_ap(class_name, dist_th, ap)
            aps.append(ap)
            ap_strs.append(f"{ap:.4f}")

        # Mean AP across distance thresholds
        mean_ap = sum(aps) / len(aps) if aps else 0.0

        # Print per-class results
        print(f"{class_name:<30s} {ap_strs[0]:<10s} {ap_strs[1]:<10s} {ap_strs[2]:<10s} {ap_strs[3]:<10s} {mean_ap:.4f}")

        # Compute TP metrics (for reference, not needed for mAP)
        for metric_name in TP_METRICS:
            metric_data = metric_data_list[(class_name, cfg.dist_th_tp)]
            # Skip unsupported metrics for certain classes
            if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                tp = float('nan')
            elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                tp = float('nan')
            else:
                tp = calc_tp(metric_data, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)

    print("-" * 80)

    # Calculate overall mAP
    metrics_summary = metrics.serialize()
    overall_map = metrics_summary['mean_ap']

    print()
    print(f"Overall mAP: {overall_map:.4f}")
    print()

    # Save metrics to file
    output_file = os.path.join(output_dir, "nuscenes_metrics_summary.json")
    print(f"Saving metrics to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    # Save detailed metrics
    details_file = os.path.join(output_dir, "nuscenes_metrics_details.json")
    with open(details_file, 'w') as f:
        json.dump(metric_data_list.serialize(), f, indent=2)

    # Save text summary
    summary_file = os.path.join(output_dir, "nuscenes_evaluation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("nuScenes Official mAP Evaluation Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Prediction file: {pred_file}\n")
        f.write(f"Sample tokens: {sample_tokens_file}\n")
        f.write(f"Data root: {dataroot}\n")
        f.write(f"Version: {version}\n")
        f.write(f"Config: {config_name}\n")
        if center_angle is not None and angle_width is not None:
            f.write(f"Angle filter: center={center_angle}°, width={angle_width}°\n")
        f.write("\n")
        f.write(f"Overall mAP: {overall_map:.4f}\n")
        f.write("\n")
        f.write("Per-class results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<30s} {'AP@0.5m':<10s} {'AP@1.0m':<10s} {'AP@2.0m':<10s} {'AP@4.0m':<10s} {'mAP':<10s}\n")
        f.write("-" * 80 + "\n")

        for class_name in cfg.class_names:
            aps = []
            ap_strs = []
            for dist_th in cfg.dist_ths:
                ap = metrics_summary['label_aps'][class_name][str(dist_th)]
                aps.append(ap)
                ap_strs.append(f"{ap:.4f}")
            mean_ap = sum(aps) / len(aps) if aps else 0.0
            f.write(f"{class_name:<30s} {ap_strs[0]:<10s} {ap_strs[1]:<10s} {ap_strs[2]:<10s} {ap_strs[3]:<10s} {mean_ap:.4f}\n")

    print(f"Saved summary to {summary_file}")
    print("=" * 80)
    print("Evaluation complete!")
    print("=" * 80)

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(
        description="Calculate mAP using official nuScenes evaluation"
    )
    parser.add_argument(
        "predictions_json",
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--sample-tokens",
        required=True,
        help="Path to sample tokens JSON file"
    )
    parser.add_argument(
        "--dataroot",
        required=True,
        help="nuScenes data root directory"
    )
    parser.add_argument(
        "--version",
        default="v1.0-mini",
        help="nuScenes version (default: v1.0-mini)"
    )
    parser.add_argument(
        "--config",
        default="detection_cvpr_2019",
        choices=["detection_cvpr_2019"],
        help="Evaluation config to use (default: detection_cvpr_2019)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for metrics (default: same as prediction file)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--center-angle",
        type=float,
        default=None,
        help="Center angle in degrees (sensor coords: 0°=forward, 90°=left, -90°=right, 180°=backward)"
    )
    parser.add_argument(
        "--angle-width",
        type=float,
        default=None,
        help="Total width of angle range in degrees (e.g., 90 for ±45° from center)"
    )

    args = parser.parse_args()

    evaluate_map_nuscenes_official(
        pred_file=args.predictions_json,
        sample_tokens_file=args.sample_tokens,
        dataroot=args.dataroot,
        version=args.version,
        config_name=args.config,
        output_dir=args.output_dir,
        center_angle=args.center_angle,
        angle_width=args.angle_width,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
