"""
Detailed nuScenes-style mAP evaluation with precision-recall curves and analysis.

This script provides comprehensive evaluation following nuScenes methodology:
- Precision-recall curve calculation and visualization
- Per-threshold detailed metrics
- Confidence score distribution analysis
- Distance error analysis

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
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kitti_label_parser import KITTILabelParser
from evaluate_kitti_predictions import parse_predictions

try:
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from nuscenes.eval.detection.constants import TP_METRICS
    from nuscenes.eval.detection.data_classes import (
        DetectionBox,
        DetectionConfig,
        DetectionMetricDataList,
        DetectionMetrics,
        DetectionMetricData,
    )
    from nuscenes.eval.common.config import config_factory
except ImportError as e:
    print(f"Error: nuScenes devkit not installed. Please install it:")
    print("  pip install nuscenes-devkit")
    sys.exit(1)


# Map KITTI classes to nuScenes classes
KITTI_TO_NUSCENES_CLASS = {
    'Car': 'car',
    'Pedestrian': 'pedestrian',
    'Cyclist': 'bicycle',
    'Van': 'car',
    'Truck': 'truck',
    'Person_sitting': 'pedestrian',
    'Tram': 'bus',
    'Misc': 'car',
    'DontCare': None,
}


def kitti_to_detection_box(bbox_3d: np.ndarray, score: float, class_name: str,
                           sample_token: str, velocity: Tuple[float, float] = (0.0, 0.0)) -> DetectionBox:
    """Convert KITTI-style bbox to nuScenes DetectionBox."""
    x, y, z, w, l, h, ry = bbox_3d

    translation = [float(x), float(y), float(z)]
    size = [float(w), float(l), float(h)]

    # Convert rotation angle to quaternion
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
        attribute_name=''
    )


def calculate_precision_recall_curve(metric_data: DetectionMetricData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract precision-recall curve from metric data.

    Returns:
        confidences: Array of confidence thresholds
        precisions: Array of precision values
        recalls: Array of recall values
    """
    # nuScenes' accumulate() already calculates precision and recall
    confidences = np.array(metric_data.confidence)
    precisions = np.array(metric_data.precision)
    recalls = np.array(metric_data.recall)

    return confidences, precisions, recalls


def plot_precision_recall_curves(
    results_by_threshold: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    output_path: str,
    class_name: str
):
    """
    Plot precision-recall curves for multiple distance thresholds.

    Args:
        results_by_threshold: Dict mapping distance threshold to (conf, prec, rec, ap)
        output_path: Path to save the plot
        class_name: Name of the class being evaluated
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Precision-Recall curves
    for dist_th in sorted(results_by_threshold.keys()):
        confidences, precisions, recalls, ap = results_by_threshold[dist_th]
        ax1.plot(recalls, precisions,
                label=f'{dist_th}m (AP={ap:.3f})',
                linewidth=2, marker='o', markersize=3, markevery=max(1, len(recalls)//20))

    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title(f'Precision-Recall Curves - {class_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Plot 2: AP by distance threshold
    dist_ths = sorted(results_by_threshold.keys())
    aps = [results_by_threshold[d][3] for d in dist_ths]

    ax2.bar(range(len(dist_ths)), aps, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Distance Threshold (m)', fontsize=12)
    ax2.set_ylabel('Average Precision', fontsize=12)
    ax2.set_title(f'AP by Distance Threshold - {class_name}', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(dist_ths)))
    ax2.set_xticklabels([f'{d}m' for d in dist_ths])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])

    # Add value labels on bars
    for i, ap in enumerate(aps):
        ax2.text(i, ap + 0.02, f'{ap:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved precision-recall curves to: {output_path}")


def plot_confidence_distribution(
    pred_boxes_dict: Dict[str, List[DetectionBox]],
    output_path: str,
    class_name: str
):
    """Plot confidence score distribution."""
    scores = []
    for boxes in pred_boxes_dict.values():
        scores.extend([box.detection_score for box in boxes])

    if not scores:
        print("No predictions to plot confidence distribution")
        return

    scores = np.array(scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Confidence Score Distribution - {class_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axvline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.3f}')
    ax1.axvline(np.median(scores), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.3f}')
    ax1.legend()

    # Cumulative distribution
    sorted_scores = np.sort(scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax2.plot(sorted_scores, cumulative, linewidth=2, color='steelblue')
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Cumulative Proportion', fontsize=12)
    ax2.set_title(f'Cumulative Distribution - {class_name}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    # Add percentile lines
    for percentile in [25, 50, 75, 90, 95]:
        value = np.percentile(scores, percentile)
        ax2.axvline(value, color='red', linestyle=':', alpha=0.5)
        ax2.text(value, 0.02, f'{percentile}%', rotation=90, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved confidence distribution to: {output_path}")


def evaluate_nuscenes_detailed(
    pred_file: str,
    gt_label_dir: str,
    target_class: str = 'Car',
    score_threshold: float = 0.0,
    filter_kitti_range: bool = False,
    x_range: Tuple[float, float] = (0.0, 70.0),
    y_range: Tuple[float, float] = (-40.0, 40.0),
    output_dir: str = None,
    verbose: bool = False
):
    """
    Detailed evaluation using nuScenes-style metrics.
    """
    print("=" * 80)
    print("nuScenes-Style DETAILED EVALUATION (KITTI Coordinates)")
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
        print(f"Score threshold: {score_threshold}")

    if filter_kitti_range:
        print(f"KITTI range filtering: ENABLED")
        print(f"  X range: [{x_range[0]:.1f}, {x_range[1]:.1f}] meters")
        print(f"  Y range: [{y_range[0]:.1f}, {y_range[1]:.1f}] meters")

    # Setup output directory
    if output_dir is None:
        output_dir = os.path.dirname(pred_file)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pred_file))[0]

    # Create detection config using official nuScenes config
    cfg = config_factory("detection_cvpr_2019")

    print(f"\nnuScenes configuration:")
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
        total_preds_after_range_filter = len(dets)

        # Convert to DetectionBox
        pred_boxes = []
        for det in dets:
            bbox_3d = np.array([
                det.bbox.x, det.bbox.y, det.bbox.z,
                det.bbox.w, det.bbox.l, det.bbox.h,
                det.bbox.ry
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
        gts = [g for g in gts if g.class_name == target_class]

        # Convert to DetectionBox
        gt_boxes = []
        for gt in gts:
            bbox_3d = np.array([
                gt.bbox.x, gt.bbox.y, gt.bbox.z,
                gt.bbox.w, gt.bbox.l, gt.bbox.h,
                gt.bbox.ry
            ])
            box = kitti_to_detection_box(
                bbox_3d, 1.0, nuscenes_class, sample_token
            )
            gt_boxes.append(box)

        gt_boxes_dict[sample_token] = gt_boxes
        total_gts += len(gt_boxes)

    print(f"\nFiltering statistics:")
    print(f"  Initial predictions: {total_preds_before_filter}")
    if score_threshold > 0:
        print(f"  After score filter: {total_preds_after_score_filter}")
    if filter_kitti_range:
        print(f"  After range filter: {total_preds_after_range_filter}")
    print(f"  Final predictions: {total_preds}")
    print(f"  Total ground truth: {total_gts}")

    # Create EvalBoxes from dict
    pred_boxes = EvalBoxes()
    gt_boxes = EvalBoxes()
    pred_boxes.boxes = pred_boxes_dict
    gt_boxes.boxes = gt_boxes_dict

    # Plot confidence distribution
    conf_dist_path = os.path.join(output_dir, f"{base_name}_confidence_distribution.png")
    plot_confidence_distribution(pred_boxes_dict, conf_dist_path, target_class)

    # Accumulate metrics for each distance threshold
    print(f"\nAccumulating metrics for class '{nuscenes_class}'...")

    metric_data_list = DetectionMetricDataList()
    results_by_threshold = {}

    # Collect all distance thresholds
    all_dist_ths = list(cfg.dist_ths)
    if cfg.dist_th_tp not in all_dist_ths:
        all_dist_ths.append(cfg.dist_th_tp)

    for dist_th in all_dist_ths:
        print(f"  Computing metrics for distance threshold: {dist_th}m")
        md = accumulate(
            gt_boxes,
            pred_boxes,
            nuscenes_class,
            cfg.dist_fcn_callable,
            dist_th
        )
        metric_data_list.set(nuscenes_class, dist_th, md)

        # Calculate precision-recall curve for this threshold
        if dist_th in cfg.dist_ths:
            confidences, precisions, recalls = calculate_precision_recall_curve(md)
            ap = calc_ap(md, cfg.min_recall, cfg.min_precision)
            results_by_threshold[dist_th] = (confidences, precisions, recalls, ap)

    # Plot precision-recall curves
    pr_curve_path = os.path.join(output_dir, f"{base_name}_precision_recall_curves.png")
    plot_precision_recall_curves(results_by_threshold, pr_curve_path, target_class)

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

    # Save detailed results
    output_file = os.path.join(output_dir, f"{base_name}_detailed_evaluation.txt")
    with open(output_file, 'w') as f:
        f.write("nuScenes-Style DETAILED Evaluation Results (KITTI Coordinates)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Prediction file: {pred_file}\n")
        f.write(f"GT directory: {gt_label_dir}\n")
        f.write(f"KITTI class: {target_class}\n")
        f.write(f"nuScenes class: {nuscenes_class}\n")
        f.write(f"Matching method: Center distance\n")
        if score_threshold > 0:
            f.write(f"Score threshold: {score_threshold}\n")
        if filter_kitti_range:
            f.write(f"KITTI range filter: ENABLED\n")
            f.write(f"  X range: [{x_range[0]:.1f}, {x_range[1]:.1f}] meters\n")
            f.write(f"  Y range: [{y_range[0]:.1f}, {y_range[1]:.1f}] meters\n")
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

    print(f"\nDetailed results saved to: {output_file}")
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 80)

    return {
        'mean_ap': mean_ap,
        'ap_by_distance': aps_dict,
        'tp_metrics': tp_metrics_summary
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Detailed nuScenes-style mAP evaluation with visualizations'
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
    parser.add_argument('--filter-kitti-range', action='store_true',
                        help='Filter predictions to KITTI evaluation range')
    parser.add_argument('--x-min', type=float, default=0.0,
                        help='Minimum X coordinate (forward) in meters (default: 0.0)')
    parser.add_argument('--x-max', type=float, default=70.0,
                        help='Maximum X coordinate (forward) in meters (default: 70.0)')
    parser.add_argument('--y-min', type=float, default=-40.0,
                        help='Minimum Y coordinate (lateral) in meters (default: -40.0)')
    parser.add_argument('--y-max', type=float, default=40.0,
                        help='Maximum Y coordinate (lateral) in meters (default: 40.0)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots and results (default: same as pred_file)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')

    args = parser.parse_args()

    evaluate_nuscenes_detailed(
        args.pred_file,
        args.gt_label_dir,
        args.target_class,
        args.score_threshold,
        args.filter_kitti_range,
        (args.x_min, args.x_max),
        (args.y_min, args.y_max),
        args.output_dir,
        args.verbose
    )
