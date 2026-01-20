"""
Evaluate KITTI predictions against ground truth

Parse prediction JSON and compare with KITTI ground truth labels.
"""

import json
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kitti_label_parser import KITTILabelParser
from kitti_eval import BBox3D, Detection, GroundTruth


def quaternion_to_yaw(quat: List[float]) -> float:
    """
    Convert quaternion to yaw angle (rotation around Z axis)

    Args:
        quat: [w, x, y, z] quaternion

    Returns:
        yaw angle in radians
    """
    w, x, y, z = quat
    # Yaw (rotation around Z axis)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw


def parse_predictions(json_file: str) -> Dict[str, List[Detection]]:
    """
    Parse prediction JSON file

    Args:
        json_file: Path to prediction JSON file

    Returns:
        Dictionary mapping sample_token -> List[Detection]
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    predictions = {}

    for sample_token, detections in data['results'].items():
        det_list = []

        for det in detections:
            # Extract data
            translation = det['translation']  # [x, y, z]
            size = det['size']  # [w, l, h]
            rotation = det['rotation']  # [qw, qx, qy, qz]
            score = det['detection_score']
            class_name = det['detection_name']

            # Convert class name (vehicle.car -> Car)
            if class_name == 'vehicle.car':
                kitti_class = 'Car'
            elif class_name == 'human.pedestrian':
                kitti_class = 'Pedestrian'
            elif class_name == 'vehicle.bicycle':
                kitti_class = 'Cyclist'
            else:
                kitti_class = class_name.split('.')[-1].capitalize()

            # Convert quaternion to yaw
            yaw = quaternion_to_yaw(rotation)

            # Create BBox3D
            # Prediction is in Velodyne coords: X=forward, Y=left, Z=up
            bbox = BBox3D(
                x=translation[0],
                y=translation[1],
                z=translation[2],
                w=size[0],
                l=size[1],
                h=size[2],
                ry=yaw,
                score=score,
                class_name=kitti_class
            )

            # Create Detection
            detection = Detection(
                bbox=bbox,
                score=score,
                class_name=kitti_class
            )

            det_list.append(detection)

        predictions[sample_token] = det_list

    return predictions


def calculate_iou_bev(bbox1: BBox3D, bbox2: BBox3D) -> float:
    """
    Calculate IoU in Bird's Eye View (X-Y plane in Velodyne coords)

    Uses center distance as approximation for IoU
    """
    # Calculate center distance in BEV (X-Y plane)
    dist = np.sqrt((bbox1.x - bbox2.x)**2 + (bbox1.y - bbox2.y)**2)

    # Calculate diagonal of bounding boxes
    diag1 = np.sqrt(bbox1.l**2 + bbox1.w**2) / 2
    diag2 = np.sqrt(bbox2.l**2 + bbox2.w**2) / 2

    # If boxes are far apart, IoU is 0
    if dist > (diag1 + diag2):
        return 0.0

    # Simple approximation: if centers are close, boxes likely overlap
    # IoU decreases linearly with distance
    max_dist = diag1 + diag2
    iou_approx = max(0, 1.0 - dist / max_dist)

    return iou_approx


def match_detections_to_ground_truth(
    detections: List[Detection],
    ground_truths: List[GroundTruth],
    iou_threshold: float = 0.5,
    match_class: bool = True
) -> Tuple[List[Tuple[Detection, GroundTruth]], List[Detection], List[GroundTruth]]:
    """
    Match detections to ground truth using greedy matching

    Args:
        detections: List of Detection objects
        ground_truths: List of GroundTruth objects
        iou_threshold: Minimum IoU for a match

    Returns:
        (matches, unmatched_detections, unmatched_ground_truths)
    """
    # Sort detections by score (highest first)
    sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)

    matches = []
    matched_gt_indices = set()
    unmatched_dets = []

    for det in sorted_dets:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt_indices:
                continue

            # Only match same class (if required)
            if match_class and det.class_name != gt.class_name:
                continue

            iou = calculate_iou_bev(det.bbox, gt.bbox)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            matches.append((det, ground_truths[best_gt_idx]))
            matched_gt_indices.add(best_gt_idx)
        else:
            unmatched_dets.append(det)

    unmatched_gts = [gt for i, gt in enumerate(ground_truths) if i not in matched_gt_indices]

    return matches, unmatched_dets, unmatched_gts


def evaluate_predictions(
    pred_file: str,
    gt_label_dir: str,
    score_threshold: float = 0.3,
    iou_threshold: float = 0.5
):
    """
    Evaluate predictions against ground truth

    Args:
        pred_file: Path to prediction JSON file
        gt_label_dir: Directory containing ground truth labels
        score_threshold: Minimum score for detections
        iou_threshold: IoU threshold for matching
    """
    print("=" * 80)
    print("KITTI PREDICTION EVALUATION")
    print("=" * 80)

    # Load predictions
    print(f"\nLoading predictions: {pred_file}")
    predictions = parse_predictions(pred_file)
    print(f"  Loaded {len(predictions)} samples")

    # Statistics
    stats_by_class = defaultdict(lambda: {
        'tp': 0, 'fp': 0, 'fn': 0,
        'total_pred': 0, 'total_gt': 0
    })

    # Process each sample
    for sample_token, dets in predictions.items():
        # Filter by score
        dets = [d for d in dets if d.score >= score_threshold]

        # Load ground truth
        gt_file = os.path.join(gt_label_dir, f"{sample_token}.txt")

        if not os.path.exists(gt_file):
            print(f"Warning: GT file not found for sample {sample_token}")
            continue

        # Parse GT in lidar coordinates (same as predictions)
        gts = KITTILabelParser.parse_ground_truth_file(gt_file, coordinate_system='lidar')

        # Match detections to ground truth
        # Note: Setting match_class=False for now due to class name mismatches
        matches, unmatched_dets, unmatched_gts = match_detections_to_ground_truth(
            dets, gts, iou_threshold, match_class=False
        )

        # Update statistics by class
        for det, gt in matches:
            stats_by_class[det.class_name]['tp'] += 1

        for det in unmatched_dets:
            stats_by_class[det.class_name]['fp'] += 1

        for gt in unmatched_gts:
            stats_by_class[gt.class_name]['fn'] += 1

        # Count totals
        for det in dets:
            stats_by_class[det.class_name]['total_pred'] += 1

        for gt in gts:
            stats_by_class[gt.class_name]['total_gt'] += 1

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Score threshold: {score_threshold}")
    print(f"IoU threshold: {iou_threshold}")

    for class_name in sorted(stats_by_class.keys()):
        stats = stats_by_class[class_name]
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{class_name}:")
        print(f"  Total predictions: {stats['total_pred']}")
        print(f"  Total ground truth: {stats['total_gt']}")
        print(f"  True Positives: {tp}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")

    # Overall statistics
    total_tp = sum(s['tp'] for s in stats_by_class.values())
    total_fp = sum(s['fp'] for s in stats_by_class.values())
    total_fn = sum(s['fn'] for s in stats_by_class.values())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print(f"\n{'Overall'}")
    print(f"  Precision: {overall_precision:.3f}")
    print(f"  Recall: {overall_recall:.3f}")
    print(f"  F1 Score: {overall_f1:.3f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate KITTI predictions')
    parser.add_argument('pred_file', type=str,
                        help='Path to prediction JSON file')
    parser.add_argument('--gt-label-dir', type=str,
                        default=r'D:\label_kitti\training\label_2',
                        help='Directory containing ground truth labels')
    parser.add_argument('--score-threshold', type=float, default=0.3,
                        help='Minimum detection score threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for matching')

    args = parser.parse_args()

    evaluate_predictions(
        args.pred_file,
        args.gt_label_dir,
        args.score_threshold,
        args.iou_threshold
    )
