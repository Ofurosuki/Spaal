"""
KITTI-style mAP Evaluation for Point Cloud Detection

This script evaluates 3D object detection results on point cloud data (.pcd.bin files)
using KITTI evaluation metrics.

Based on KITTI Object Detection Benchmark:
http://www.cvlibs.net/datasets/kitti/eval_object.php
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class BBox3D:
    """3D Bounding Box representation"""
    x: float  # center x
    y: float  # center y
    z: float  # center z
    w: float  # width
    l: float  # length
    h: float  # height
    ry: float  # rotation around y-axis
    score: float = 1.0
    class_name: str = ""


@dataclass
class Detection:
    """Detection result"""
    bbox: BBox3D
    score: float
    class_name: str


@dataclass
class GroundTruth:
    """Ground truth annotation"""
    bbox: BBox3D
    class_name: str
    truncation: float = 0.0
    occlusion: int = 0
    difficulty: str = "MODERATE"


class KITTIEvaluator:
    """KITTI-style 3D Object Detection Evaluator for Point Clouds"""

    # KITTI evaluation parameters
    MIN_HEIGHT = {'EASY': 40, 'MODERATE': 25, 'HARD': 25}
    MAX_OCCLUSION = {'EASY': 0, 'MODERATE': 1, 'HARD': 2}
    MAX_TRUNCATION = {'EASY': 0.15, 'MODERATE': 0.3, 'HARD': 0.5}

    # IoU thresholds for different classes (BEV and 3D)
    IOU_THRESHOLDS = {
        'Car': {'bev': 0.7, '3d': 0.7},
        'Pedestrian': {'bev': 0.5, '3d': 0.5},
        'Cyclist': {'bev': 0.5, '3d': 0.5}
    }

    # Number of recall discretization steps
    N_SAMPLE_PTS = 41

    def __init__(self, classes: List[str] = ['Car', 'Pedestrian', 'Cyclist']):
        """
        Initialize KITTI evaluator

        Args:
            classes: List of class names to evaluate
        """
        self.classes = classes

    @staticmethod
    def compute_iou_bev(bbox1: BBox3D, bbox2: BBox3D) -> float:
        """
        Compute Bird's Eye View (BEV) IoU between two 3D bounding boxes

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box

        Returns:
            IoU value (0.0 to 1.0)
        """
        # Get corners in BEV (x-z plane)
        corners1 = KITTIEvaluator._get_bev_corners(bbox1)
        corners2 = KITTIEvaluator._get_bev_corners(bbox2)

        # Compute intersection area using Sutherland-Hodgman algorithm
        intersection_area = KITTIEvaluator._polygon_intersection_area(corners1, corners2)

        # Compute union area
        area1 = bbox1.w * bbox1.l
        area2 = bbox2.w * bbox2.l
        union_area = area1 + area2 - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    @staticmethod
    def compute_iou_3d(bbox1: BBox3D, bbox2: BBox3D) -> float:
        """
        Compute 3D IoU between two 3D bounding boxes

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box

        Returns:
            IoU value (0.0 to 1.0)
        """
        # Compute BEV IoU
        bev_iou = KITTIEvaluator.compute_iou_bev(bbox1, bbox2)

        if bev_iou == 0:
            return 0.0

        # Compute height overlap
        y_min1 = bbox1.y - bbox1.h / 2
        y_max1 = bbox1.y + bbox1.h / 2
        y_min2 = bbox2.y - bbox2.h / 2
        y_max2 = bbox2.y + bbox2.h / 2

        y_overlap = max(0, min(y_max1, y_max2) - max(y_min1, y_min2))
        y_union = max(y_max1, y_max2) - min(y_min1, y_min2)

        if y_union == 0:
            return 0.0

        # 3D IoU = (BEV intersection * height overlap) / (volume1 + volume2 - BEV intersection * height overlap)
        vol1 = bbox1.w * bbox1.l * bbox1.h
        vol2 = bbox2.w * bbox2.l * bbox2.h

        bev_intersection = bev_iou * (bbox1.w * bbox1.l + bbox2.w * bbox2.l) / (1 + bev_iou)
        intersection_3d = bev_intersection * y_overlap

        union_3d = vol1 + vol2 - intersection_3d

        if union_3d == 0:
            return 0.0

        return intersection_3d / union_3d

    @staticmethod
    def _get_bev_corners(bbox: BBox3D) -> np.ndarray:
        """Get 4 corners of bounding box in BEV (bird's eye view)"""
        # Center of bbox
        cx, cz = bbox.x, bbox.z
        w, l = bbox.w, bbox.l
        ry = bbox.ry

        # 4 corners before rotation (centered at origin)
        corners = np.array([
            [-l/2, -w/2],
            [-l/2,  w/2],
            [ l/2,  w/2],
            [ l/2, -w/2]
        ])

        # Rotation matrix
        cos_ry = np.cos(ry)
        sin_ry = np.sin(ry)
        R = np.array([
            [cos_ry, -sin_ry],
            [sin_ry,  cos_ry]
        ])

        # Rotate and translate
        corners = corners @ R.T
        corners[:, 0] += cx
        corners[:, 1] += cz

        return corners

    @staticmethod
    def _polygon_intersection_area(poly1: np.ndarray, poly2: np.ndarray) -> float:
        """
        Compute intersection area of two convex polygons using Sutherland-Hodgman algorithm

        Simplified version - assumes axis-aligned or small rotation
        """
        from shapely.geometry import Polygon

        p1 = Polygon(poly1)
        p2 = Polygon(poly2)

        if not p1.is_valid or not p2.is_valid:
            return 0.0

        intersection = p1.intersection(p2)
        return intersection.area

    def compute_precision_recall(
        self,
        detections: List[Detection],
        ground_truths: List[GroundTruth],
        class_name: str,
        difficulty: str = 'MODERATE',
        metric: str = 'bev'
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute precision-recall curve for a single class

        Args:
            detections: List of detections
            ground_truths: List of ground truth annotations
            class_name: Class name to evaluate
            difficulty: Difficulty level ('EASY', 'MODERATE', 'HARD')
            metric: Evaluation metric ('bev' or '3d')

        Returns:
            precision: Precision array
            recall: Recall array
            ap: Average Precision
        """
        # Filter by class
        dets = [d for d in detections if d.class_name == class_name]
        gts = [g for g in ground_truths if g.class_name == class_name]

        if len(gts) == 0:
            return np.zeros(self.N_SAMPLE_PTS), np.zeros(self.N_SAMPLE_PTS), 0.0

        # Sort detections by score (descending)
        dets = sorted(dets, key=lambda x: x.score, reverse=True)

        # Get IoU threshold
        iou_threshold = self.IOU_THRESHOLDS[class_name]['bev' if metric == 'bev' else '3d']

        # Match detections to ground truths
        num_gts = len(gts)
        matched_gts = [False] * num_gts

        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))

        for i, det in enumerate(dets):
            max_iou = 0.0
            max_gt_idx = -1

            # Find best matching ground truth
            for j, gt in enumerate(gts):
                if matched_gts[j]:
                    continue

                if metric == 'bev':
                    iou = self.compute_iou_bev(det.bbox, gt.bbox)
                else:
                    iou = self.compute_iou_3d(det.bbox, gt.bbox)

                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = j

            # Check if detection matches
            if max_iou >= iou_threshold:
                matched_gts[max_gt_idx] = True
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        # Compute cumulative TP and FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # Compute precision and recall
        recalls = tp_cumsum / num_gts
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        # Interpolate precision-recall curve
        recall_samples = np.linspace(0, 1, self.N_SAMPLE_PTS)
        precision_interp = np.zeros(self.N_SAMPLE_PTS)

        for i, r in enumerate(recall_samples):
            # Find all precisions where recall >= r
            valid_precisions = precisions[recalls >= r]
            if len(valid_precisions) > 0:
                precision_interp[i] = np.max(valid_precisions)
            else:
                precision_interp[i] = 0.0

        # Compute AP (average over recall range 0.1 to 1.0, following KITTI protocol)
        # AP = sum of precision values from index 1 to 40 / 40
        ap = np.mean(precision_interp[1:])

        return precision_interp, recall_samples, ap

    def evaluate(
        self,
        detections: List[Detection],
        ground_truths: List[GroundTruth],
        difficulties: List[str] = ['EASY', 'MODERATE', 'HARD'],
        metrics: List[str] = ['bev', '3d']
    ) -> Dict:
        """
        Evaluate detections against ground truths

        Args:
            detections: List of all detections
            ground_truths: List of all ground truth annotations
            difficulties: List of difficulty levels to evaluate
            metrics: List of metrics to compute ('bev', '3d')

        Returns:
            results: Dictionary with evaluation results
        """
        results = {}

        for class_name in self.classes:
            results[class_name] = {}

            for difficulty in difficulties:
                results[class_name][difficulty] = {}

                for metric in metrics:
                    precision, recall, ap = self.compute_precision_recall(
                        detections, ground_truths, class_name, difficulty, metric
                    )

                    results[class_name][difficulty][metric] = {
                        'precision': precision,
                        'recall': recall,
                        'ap': ap
                    }

        # Compute mAP
        for difficulty in difficulties:
            for metric in metrics:
                aps = [results[cls][difficulty][metric]['ap'] for cls in self.classes]
                results[f'mAP_{metric}_{difficulty}'] = np.mean(aps)

        return results


def load_pcd_bin(filepath: str) -> np.ndarray:
    """
    Load point cloud from .pcd.bin file

    Args:
        filepath: Path to .pcd.bin file

    Returns:
        points: Nx5 array (x, y, z, intensity, timestamp)
    """
    points = np.fromfile(filepath, dtype=np.float32)
    points = points.reshape(-1, 5)
    return points


def print_results(results: Dict):
    """Print evaluation results in a formatted table"""
    print("\n" + "="*80)
    print("KITTI-STYLE EVALUATION RESULTS")
    print("="*80)

    # Print per-class results
    for class_name in ['Car', 'Pedestrian', 'Cyclist']:
        if class_name not in results:
            continue

        print(f"\n{class_name}:")
        print("-" * 80)
        print(f"{'Difficulty':<12} {'BEV AP':<15} {'3D AP':<15}")
        print("-" * 80)

        for difficulty in ['EASY', 'MODERATE', 'HARD']:
            if difficulty not in results[class_name]:
                continue

            bev_ap = results[class_name][difficulty].get('bev', {}).get('ap', 0.0) * 100
            d3_ap = results[class_name][difficulty].get('3d', {}).get('ap', 0.0) * 100

            print(f"{difficulty:<12} {bev_ap:>6.2f}%        {d3_ap:>6.2f}%")

    # Print mAP
    print("\n" + "="*80)
    print("MEAN AVERAGE PRECISION (mAP)")
    print("="*80)
    print(f"{'Difficulty':<12} {'BEV mAP':<15} {'3D mAP':<15}")
    print("-" * 80)

    for difficulty in ['EASY', 'MODERATE', 'HARD']:
        bev_map = results.get(f'mAP_bev_{difficulty}', 0.0) * 100
        d3_map = results.get(f'mAP_3d_{difficulty}', 0.0) * 100
        print(f"{difficulty:<12} {bev_map:>6.2f}%        {d3_map:>6.2f}%")

    print("="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    print("KITTI-style mAP Evaluation Module")
    print("Usage: Import this module and use KITTIEvaluator class")
    print("\nRequired: shapely library for polygon intersection")
    print("Install: pip install shapely")
