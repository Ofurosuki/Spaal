"""
Calculate mAP following KITTI official C++ implementation

Reference: D:/kitti_evaluation/cpp/evaluate_object.cpp
"""

import json
import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kitti_label_parser import KITTILabelParser
from evaluate_kitti_predictions import parse_predictions, calculate_iou_bev
from kitti_eval import Detection, GroundTruth


# KITTI official parameters
N_SAMPLE_PTS = 41  # number of recall points (KITTI uses 41)
MIN_OVERLAP_CAR = 0.7  # minimum IoU for Car class


def get_thresholds(scores: List[float], n_groundtruth: int) -> List[float]:
    """
    Get score thresholds for N_SAMPLE_PTS recall values
    Following KITTI official implementation (lines 349-381)

    Args:
        scores: Detection scores (will be sorted descending)
        n_groundtruth: Total number of ground truth objects

    Returns:
        List of score thresholds
    """
    # Sort scores in descending order
    v = sorted(scores, reverse=True)

    thresholds = []
    current_recall = 0.0

    for i in range(len(v)):
        # Calculate left and right recall
        l_recall = (i + 1) / n_groundtruth
        if i < len(v) - 1:
            r_recall = (i + 2) / n_groundtruth
        else:
            r_recall = l_recall

        # Check if right-hand-side recall is closer than left-hand-side
        if (r_recall - current_recall) < (current_recall - l_recall) and i < len(v) - 1:
            continue

        # Use this score threshold
        thresholds.append(v[i])
        current_recall += 1.0 / (N_SAMPLE_PTS - 1.0)

        # Stop if we have enough thresholds
        if len(thresholds) >= N_SAMPLE_PTS:
            break

    return thresholds


def compute_statistics_at_threshold(
    detections: List[Detection],
    ground_truths: List[GroundTruth],
    threshold: float,
    min_overlap: float
) -> Tuple[int, int, int, List[float]]:
    """
    Compute TP, FP, FN at a given score threshold
    Following KITTI official implementation (lines 459-600)

    Args:
        detections: List of detections
        ground_truths: List of ground truth objects
        threshold: Score threshold
        min_overlap: Minimum IoU for matching

    Returns:
        (tp, fp, fn, detection_scores)
    """
    tp = 0
    fp = 0
    fn = 0
    detection_scores = []

    # Filter detections by threshold
    valid_dets = [d for d in detections if d.score >= threshold]

    # Track which detections and GTs are assigned
    assigned_detection = [False] * len(valid_dets)
    assigned_gt = [False] * len(ground_truths)

    # For each ground truth, find best matching detection
    for gt_idx, gt in enumerate(ground_truths):
        best_det_idx = -1
        max_overlap = 0.0

        for det_idx, det in enumerate(valid_dets):
            if assigned_detection[det_idx]:
                continue

            overlap = calculate_iou_bev(det.bbox, gt.bbox)

            if overlap > max_overlap:
                max_overlap = overlap
                best_det_idx = det_idx

        # Check if match found
        if max_overlap >= min_overlap:
            # True positive
            tp += 1
            assigned_detection[best_det_idx] = True
            assigned_gt[gt_idx] = True
            detection_scores.append(valid_dets[best_det_idx].score)
        else:
            # False negative (no detection for this GT)
            fn += 1

    # Count false positives (unassigned detections)
    fp = sum(1 for assigned in assigned_detection if not assigned)

    return tp, fp, fn, detection_scores


def evaluate_class(
    all_detections: List[List[Detection]],
    all_ground_truths: List[List[GroundTruth]],
    min_overlap: float = MIN_OVERLAP_CAR,
    difficulty: str = None
) -> Tuple[List[float], List[float], float]:
    """
    Evaluate a single class following KITTI official implementation
    (lines 627-709)

    Args:
        all_detections: List of detection lists (one per sample)
        all_ground_truths: List of GT lists (one per sample)
        min_overlap: Minimum IoU for matching
        difficulty: Filter GT by difficulty ('EASY', 'MODERATE', 'HARD', or None for all)

    Returns:
        (precision, recall, AP)
    """
    # Filter ground truth by difficulty if specified
    if difficulty:
        filtered_gts = []
        for gts in all_ground_truths:
            filtered = [gt for gt in gts if gt.difficulty == difficulty]
            filtered_gts.append(filtered)
        all_ground_truths = filtered_gts

    # Count total ground truth
    n_gt = sum(len(gts) for gts in all_ground_truths)

    if n_gt == 0:
        print("Warning: No ground truth objects found")
        return [], [], 0.0

    # Collect all detection scores
    all_scores = []
    for dets in all_detections:
        all_scores.extend([d.score for d in dets])

    if len(all_scores) == 0:
        print("Warning: No detections found")
        return [0.0] * N_SAMPLE_PTS, [0.0] * N_SAMPLE_PTS, 0.0

    # Get score thresholds for recall discretization
    thresholds = get_thresholds(all_scores, n_gt)

    print(f"\nComputed {len(thresholds)} thresholds for {N_SAMPLE_PTS} recall points")
    print(f"Score range: [{min(thresholds):.4f}, {max(thresholds):.4f}]")

    # Compute statistics at each threshold
    pr_data = []
    for thresh in thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for dets, gts in zip(all_detections, all_ground_truths):
            tp, fp, fn, _ = compute_statistics_at_threshold(dets, gts, thresh, min_overlap)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        pr_data.append({
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'threshold': thresh
        })

    # Compute precision and recall
    precision = []
    recall = []

    for pr in pr_data:
        r = pr['tp'] / (pr['tp'] + pr['fn']) if (pr['tp'] + pr['fn']) > 0 else 0.0
        p = pr['tp'] / (pr['tp'] + pr['fp']) if (pr['tp'] + pr['fp']) > 0 else 0.0

        recall.append(r)
        precision.append(p)

    # Pad to N_SAMPLE_PTS if needed
    while len(precision) < N_SAMPLE_PTS:
        precision.append(0.0)
        recall.append(0.0)

    # Smooth precision using max_{i..end}(precision)
    # This is the key step in KITTI evaluation (line 698-699)
    for i in range(len(precision)):
        precision[i] = max(precision[i:])

    # Calculate AP as average of precisions
    ap = np.mean(precision[:N_SAMPLE_PTS])

    return precision, recall, ap


def evaluate_map_official(
    pred_file: str,
    gt_label_dir: str,
    target_class: str = 'Car',
    min_overlap: float = MIN_OVERLAP_CAR
):
    """
    Evaluate mAP following KITTI official implementation

    Args:
        pred_file: Path to prediction JSON file
        gt_label_dir: Directory containing ground truth labels
        target_class: Class to evaluate
        min_overlap: Minimum IoU for matching
    """
    print("=" * 80)
    print("KITTI OFFICIAL mAP EVALUATION")
    print("=" * 80)
    print(f"Target class: {target_class}")
    print(f"Minimum overlap (IoU): {min_overlap}")
    print(f"Recall sampling points: {N_SAMPLE_PTS}")

    # Load predictions
    print(f"\nLoading predictions: {pred_file}")
    predictions = parse_predictions(pred_file)
    print(f"  Loaded {len(predictions)} samples")

    # Collect detections and ground truths per sample
    all_detections = []
    all_ground_truths = []
    total_dets = 0
    total_gts = 0

    for sample_idx, (sample_token, dets) in enumerate(predictions.items()):
        # Filter by class
        dets = [d for d in dets if d.class_name == target_class]
        total_dets += len(dets)

        # Load ground truth
        gt_file = os.path.join(gt_label_dir, f"{sample_token}.txt")

        if not os.path.exists(gt_file):
            print(f"Warning: GT file not found for sample {sample_token}")
            continue

        # Parse GT in lidar coordinates
        gts = KITTILabelParser.parse_ground_truth_file(gt_file, coordinate_system='lidar')

        # Filter GT by class
        gts = [g for g in gts if g.class_name == target_class]
        total_gts += len(gts)

        all_detections.append(dets)
        all_ground_truths.append(gts)

        # Progress
        if (sample_idx + 1) % 20 == 0:
            print(f"  Processed {sample_idx + 1}/{len(predictions)} samples...")

    print(f"\nTotal detections: {total_dets}")
    print(f"Total ground truth: {total_gts}")

    # Count GT by difficulty
    difficulty_counts = {'EASY': 0, 'MODERATE': 0, 'HARD': 0}
    for gts in all_ground_truths:
        for gt in gts:
            difficulty_counts[gt.difficulty] += 1

    print(f"\nGround truth by difficulty:")
    print(f"  Easy: {difficulty_counts['EASY']}")
    print(f"  Moderate: {difficulty_counts['MODERATE']}")
    print(f"  Hard: {difficulty_counts['HARD']}")

    # Evaluate ALL difficulties together first
    print("\n" + "=" * 80)
    print("EVALUATING ALL DIFFICULTIES TOGETHER")
    print("=" * 80)
    precision_all, recall_all, ap_all = evaluate_class(
        all_detections, all_ground_truths, min_overlap, difficulty=None
    )
    print(f"  AP@{min_overlap} (ALL): {ap_all * 100:.2f}%")

    # Evaluate for each difficulty level
    results = {}
    results['ALL'] = {
        'precision': precision_all,
        'recall': recall_all,
        'ap': ap_all
    }

    print("\n" + "=" * 80)
    print("EVALUATING BY DIFFICULTY")
    print("=" * 80)

    for difficulty in ['EASY', 'MODERATE', 'HARD']:
        print(f"\n--- {difficulty} ---")
        precision, recall, ap = evaluate_class(
            all_detections, all_ground_truths, min_overlap, difficulty
        )
        results[difficulty] = {
            'precision': precision,
            'recall': recall,
            'ap': ap
        }
        print(f"  AP@{min_overlap}: {ap * 100:.2f}%")

    # Print overall results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nClass: {target_class}")
    print(f"Minimum overlap (IoU): {min_overlap}")
    print(f"\n{'Difficulty':<15} {'AP':<15} {'AP%':<15}")
    print("-" * 45)
    for difficulty in ['ALL', 'EASY', 'MODERATE', 'HARD']:
        ap = results[difficulty]['ap']
        print(f"{difficulty:<15} {ap:<15.4f} {ap*100:<15.2f}")

    # Save results
    output_file = pred_file.replace('.json', '_map_official.txt')
    with open(output_file, 'w') as f:
        f.write(f"KITTI Official mAP Evaluation Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Prediction file: {pred_file}\n")
        f.write(f"GT directory: {gt_label_dir}\n")
        f.write(f"Target class: {target_class}\n")
        f.write(f"Minimum overlap: {min_overlap}\n")
        f.write(f"Recall points: {N_SAMPLE_PTS}\n\n")
        f.write(f"Total detections: {total_dets}\n")
        f.write(f"Total ground truth: {total_gts}\n\n")
        f.write(f"Ground truth by difficulty:\n")
        f.write(f"  Easy: {difficulty_counts['EASY']}\n")
        f.write(f"  Moderate: {difficulty_counts['MODERATE']}\n")
        f.write(f"  Hard: {difficulty_counts['HARD']}\n\n")
        f.write(f"Results by Difficulty:\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'Difficulty':<15} {'AP':<15} {'AP%':<15}\n")
        f.write("-" * 45 + "\n")
        for difficulty in ['ALL', 'EASY', 'MODERATE', 'HARD']:
            ap_val = results[difficulty]['ap']
            f.write(f"{difficulty:<15} {ap_val:<15.4f} {ap_val*100:<15.2f}\n")

        # Write precision-recall curves for each difficulty
        for difficulty in ['ALL', 'EASY', 'MODERATE', 'HARD']:
            f.write(f"\n\nPrecision-Recall Curve ({difficulty}):\n")
            f.write(f"{'Recall':<15} {'Precision':<15}\n")
            f.write("-" * 30 + "\n")
            for r, p in zip(results[difficulty]['recall'], results[difficulty]['precision']):
                f.write(f"{r:<15.4f} {p:<15.4f}\n")

    print(f"\nResults saved to: {output_file}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Calculate mAP following KITTI official implementation')
    parser.add_argument('pred_file', type=str,
                        help='Path to prediction JSON file')
    parser.add_argument('--gt-label-dir', type=str,
                        default=r'D:\label_kitti\training\label_2',
                        help='Directory containing ground truth labels')
    parser.add_argument('--class', dest='target_class', type=str, default='Car',
                        help='Target class to evaluate')
    parser.add_argument('--min-overlap', type=float, default=MIN_OVERLAP_CAR,
                        help='Minimum IoU for matching (KITTI uses 0.7 for Car)')

    args = parser.parse_args()

    evaluate_map_official(
        args.pred_file,
        args.gt_label_dir,
        args.target_class,
        args.min_overlap
    )
