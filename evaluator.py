import json
import numpy as np
import argparse
from collections import defaultdict
from shapely.geometry import Polygon
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import os

# Matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# nuScenes official distance thresholds in meters
NUSCENES_DIST_THRESHOLDS = {
    'vehicle.car': 2,
    'vehicle.truck': 2,
    'vehicle.bus': 2,
    'vehicle.trailer': 2,
    'vehicle.construction': 2,
    'human.pedestrian': 1,
    'vehicle.motorcycle': 1,
    'vehicle.bicycle': 1,
    'movable_object.trafficcone': 1,
    'movable_object.barrier': 1,
    'movable_object.pushable_pullable': 1
}
DEFAULT_DIST_THRESHOLD = 2 # Default threshold for classes not in the map

# --- Helper Functions ---

def get_general_class_name(full_name: str) -> str:
    """
    Converts a detailed class name like 'human.pedestrian.adult' to a general one like 'human.pedestrian'.
    """
    parts = full_name.split('.')
    if len(parts) > 2:
        return '.'.join(parts[:2])
    return full_name

def get_bev_box_corners(box: Box):
    """Calculates the 4 corners of a 2D BEV box from a Box object."""
    center_x, center_y = box.center[:2]
    width, length = box.wlh[:2]
    yaw = box.orientation.yaw_pitch_roll[0]

    half_l, half_w = length / 2, width / 2
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    corners = np.array([
        [-half_l, -half_w], [-half_l, half_w],
        [half_l, half_w], [half_l, -half_w]
    ])

    rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    transformed_corners = corners @ rot_matrix.T + np.array([center_x, center_y])
    return transformed_corners

def bev_iou(box1_corners, box2_corners):
    """Calculates IoU for two BEV boxes given their corners."""
    poly1 = Polygon(box1_corners)
    poly2 = Polygon(box2_corners)

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def is_box_in_fov(box: Box, fov_center: float, fov_width: float) -> bool:
    """Checks if a box center is within the specified horizontal FOV."""
    box_angle_rad = np.arctan2(box.center[1], box.center[0])
    box_angle_deg = np.rad2deg(box_angle_rad)

    target_angle = (box_angle_deg + 360) % 360
    
    half_width = fov_width / 2
    start_angle = (fov_center - half_width + 360) % 360
    end_angle = (fov_center + half_width + 360) % 360

    if start_angle <= end_angle:
        return start_angle <= target_angle <= end_angle
    else:  # FOV wraps around 360 degrees
        return target_angle >= start_angle or target_angle <= end_angle

def plot_bev_sample(gt_boxes_by_class, pred_boxes_by_class, pred_matched_masks_by_class, sample_token, output_dir, fov_center, fov_width):
    """
    Plots all GT and prediction boxes for a single sample in BEV.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    for class_name, gt_boxes in gt_boxes_by_class.items():
        if class_name.startswith('human'):
            color = 'cyan'
            label = 'GT (Human)'
        else:
            color = 'green'
            label = 'GT (Other)'

        for gt_box in gt_boxes:
            corners = get_bev_box_corners(gt_box)
            ax.add_patch(MplPolygon(corners, closed=True, color=color, fill=False, linewidth=2, label=label))

    for class_name, pred_boxes in pred_boxes_by_class.items():
        matched_mask = pred_matched_masks_by_class.get(class_name, [False]*len(pred_boxes))
        for i, pred_box in enumerate(pred_boxes):
            corners = get_bev_box_corners(pred_box)
            is_matched = matched_mask[i]
            color = 'blue' if is_matched else 'red'
            linestyle = 'solid' if is_matched else 'dashed'
            label = 'Prediction (Matched)' if is_matched else 'Prediction (Unmatched)'
            ax.add_patch(MplPolygon(corners, closed=True, color=color, fill=False, linewidth=1, linestyle=linestyle, label=label))

    if fov_center is not None and fov_width is not None:
        line_length = 100.0
        half_width_rad = np.deg2rad(fov_width / 2)
        center_rad = np.deg2rad(fov_center)
        
        start_angle_rad = center_rad - half_width_rad
        end_angle_rad = center_rad + half_width_rad
        start_x, start_y = line_length * np.cos(start_angle_rad), line_length * np.sin(start_angle_rad)
        ax.plot([0, start_x], [0, start_y], 'y--', linewidth=2, label='FOV Boundary')

        end_x, end_y = line_length * np.cos(end_angle_rad), line_length * np.sin(end_angle_rad)
        ax.plot([0, end_x], [0, end_y], 'y--', linewidth=2)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'BEV Matching for Sample: {sample_token}')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.grid(True)
    plot_path = os.path.join(output_dir, f'{sample_token}_full.png')
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"    -> Saved full sample BEV plot to {plot_path}")

class CustommAPEvaluator:
    def __init__(self, input_path: str, dataroot: str, version: str = 'v1.0-mini', plot_dir: str = None, score_threshold: float = 0.1):
        self.input_path = input_path
        self.dataroot = dataroot
        self.version = version
        self.plot_dir = plot_dir
        self.score_threshold = score_threshold
        self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)

    def _load_data(self):
        print(f"Loading predictions from {self.input_path}...")
        all_results = {}
        if os.path.isfile(self.input_path):
            try:
                with open(self.input_path) as f:
                    all_results = json.load(f)['results']
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[ERROR] Failed to load or parse prediction file: {e}")
                return False
        else:
            print(f"[ERROR] Input path not found: {self.input_path}")
            return False

        self.pred_boxes_by_token = defaultdict(list)
        for s_token, p_boxes in all_results.items():
            for p_box in p_boxes:
                try:
                    name = str(p_box['detection_name'])
                    score = float(p_box['detection_score'])
                    if score <= self.score_threshold:
                        continue
                    size_json = p_box['size']
                    #size_for_box = [size_json[1], size_json[0], size_json[2]]
                    size_for_box = [size_json[0], size_json[1], size_json[2]]
                    box = Box(center=p_box['translation'],
                    size=size_for_box,
                    orientation=Quaternion(p_box['rotation']))
                    #rot_90_deg = Quaternion(axis=[0, 0, 1], angle=np.pi / 2)
                    #box.orientation = rot_90_deg * box.orientation
                    box.label = name
                    box.score = score
                    self.pred_boxes_by_token[s_token].append(box)
                except (KeyError, ValueError) as e:
                    continue

        self.sample_tokens = list(self.pred_boxes_by_token.keys())
        print("Loading and transforming ground truth to sensor frame...")
        self.gt_boxes_by_token = defaultdict(list)
        for s_token in self.sample_tokens:
            try:
                sample = self.nusc.get('sample', s_token)
            except KeyError:
                print(f"[WARNING] Sample token {s_token} not found in {self.version} dataset. Skipping.")
                continue
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_sd_record = self.nusc.get('sample_data', lidar_token)
            cs_record = self.nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
            ep_record = self.nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])
            for ann_token in sample['anns']:
                gt_box_global = self.nusc.get_box(ann_token)
                ann_record = self.nusc.get('sample_annotation', ann_token)
                gt_box_global.label = ann_record['category_name']
                gt_box_global.translate(-np.array(ep_record['translation']))
                gt_box_global.rotate(Quaternion(ep_record['rotation']).inverse)
                gt_box_global.translate(-np.array(cs_record['translation']))
                gt_box_global.rotate(Quaternion(cs_record['rotation']).inverse)
                self.gt_boxes_by_token[s_token].append(gt_box_global)
        
        # Filter out samples that were not found in the dataset
        self.sample_tokens = [t for t in self.sample_tokens if t in self.gt_boxes_by_token]
        print(f"Loaded data for {len(self.sample_tokens)} samples.")
        return True

    def _calculate_ap(self, class_results, total_gt_for_class):
        if total_gt_for_class == 0 or not class_results:
            return 0.0
        class_results.sort(key=lambda x: x['score'], reverse=True)
        is_tp = np.array([r['is_tp'] for r in class_results])
        tp_cumulative = np.cumsum(is_tp)
        fp_cumulative = np.cumsum(~is_tp)
        recalls = tp_cumulative / total_gt_for_class
        precisions = tp_cumulative / (tp_cumulative + fp_cumulative)
        interpolated_precisions = np.copy(precisions)
        for i in range(len(precisions) - 2, -1, -1):
            interpolated_precisions[i] = max(interpolated_precisions[i], interpolated_precisions[i+1])
        indices = np.where((recalls > 0.1) & (interpolated_precisions > 0.1))[0]
        if len(indices) < 2:
            return 0.0
        filtered_recalls = recalls[indices]
        filtered_precisions = interpolated_precisions[indices]
        ap = np.trapz(y=filtered_precisions, x=filtered_recalls)
        return ap

    def evaluate(self, iou_threshold: float = 0.5, fov_center: float = None, fov_width: float = None, match_metric: str = 'iou'):
        if not self._load_data():
            print("Halting evaluation due to data loading failure."); return
        if match_metric == 'dist':
            return self._evaluate_by_dist(fov_center, fov_width)
        else:
            return self._evaluate_by_iou(iou_threshold, fov_center, fov_width)

    def _evaluate_by_iou(self, iou_threshold, fov_center, fov_width):
        all_class_results, total_gts_by_class = self._perform_matching('iou', iou_threshold, fov_center, fov_width)
        mAP, ap_by_class = self._calculate_final_map(all_class_results, total_gts_by_class)
        print(f"\n--- Custom Evaluation Results (Sensor Frame) ---")
        print(f"mAP @ {iou_threshold} IoU: {mAP:.4f}")
        print("\nPer-Class AP:")
        for class_name, ap in sorted(ap_by_class.items()):
            print(f"- {class_name}: {ap:.4f}")
        return mAP

    def _evaluate_by_dist(self, fov_center, fov_width):
        dist_thresholds_to_test = [0.5, 1, 2, 4]
        mAPs_at_thresholds = []
        final_ap_by_class = defaultdict(float)
        for dist_thresh in dist_thresholds_to_test:
            print(f"\n--- Calculating for distance threshold: {dist_thresh}m ---")
            all_class_results, total_gts_by_class = self._perform_matching('dist', dist_thresh, fov_center, fov_width)
            mAP_for_thresh, ap_by_class = self._calculate_final_map(all_class_results, total_gts_by_class)
            print(f"mAP @ {dist_thresh}m: {mAP_for_thresh:.4f}")
            mAPs_at_thresholds.append(mAP_for_thresh)
            for class_name, ap in ap_by_class.items():
                final_ap_by_class[class_name] += ap
        final_mAP = np.mean(mAPs_at_thresholds)
        print(f"\n--- Final Custom Evaluation Result ---")
        print(f"Mean AP over thresholds {dist_thresholds_to_test}m: {final_mAP:.4f}")
        print("\nMean Per-Class AP:")
        for class_name in sorted(final_ap_by_class.keys()):
            mean_ap = final_ap_by_class[class_name] / len(dist_thresholds_to_test)
            print(f"- {class_name}: {mean_ap:.4f}")
        return final_mAP

    def _perform_matching(self, metric, threshold, fov_center, fov_width):
        all_class_results = defaultdict(list)
        print("\nMatching predictions to ground truth...")
        for s_token in self.sample_tokens:
            preds = self.pred_boxes_by_token[s_token]
            gts = self.gt_boxes_by_token[s_token]
            if fov_center is not None and fov_width is not None:
                preds = [box for box in preds if is_box_in_fov(box, fov_center, fov_width)]
                gts = [box for box in gts if is_box_in_fov(box, fov_center, fov_width)]
            preds_by_class = defaultdict(list); gts_by_class = defaultdict(list)
            for box in preds: preds_by_class[get_general_class_name(box.label)].append(box)
            for box in gts: gts_by_class[get_general_class_name(box.label)].append(box)
            all_class_names_sample = gts_by_class.keys() | preds_by_class.keys()
            pred_matched_masks_by_class = {}
            for class_name in all_class_names_sample:
                sample_preds = sorted(preds_by_class.get(class_name, []), key=lambda x: x.score, reverse=True)
                sample_gts = gts_by_class.get(class_name, [])
                gt_matched = [False] * len(sample_gts)
                pred_matched_mask = []
                for pred in sample_preds:
                    if not sample_gts:
                        all_class_results[class_name].append({'score': pred.score, 'is_tp': False}); pred_matched_mask.append(False); continue
                    best_gt_idx = -1
                    if metric == 'iou':
                        pred_corners = get_bev_box_corners(pred)
                        ious = [bev_iou(pred_corners, get_bev_box_corners(gt)) if not gt_matched[i] else -1 for i, gt in enumerate(sample_gts)]
                        if len(ious) > 0 and np.max(ious) > threshold: best_gt_idx = np.argmax(ious)
                    elif metric == 'dist':
                        distances = [np.linalg.norm(pred.center[:2] - gt.center[:2]) if not gt_matched[i] else np.inf for i, gt in enumerate(sample_gts)]
                        if len(distances) > 0 and np.min(distances) < threshold: best_gt_idx = np.argmin(distances)
                    if best_gt_idx != -1:
                        all_class_results[class_name].append({'score': pred.score, 'is_tp': True}); gt_matched[best_gt_idx] = True; pred_matched_mask.append(True)
                    else:
                        all_class_results[class_name].append({'score': pred.score, 'is_tp': False}); pred_matched_mask.append(False)
                pred_matched_masks_by_class[class_name] = pred_matched_mask
            if self.plot_dir:
                plot_bev_sample(gts_by_class, preds_by_class, pred_matched_masks_by_class, s_token, self.plot_dir, fov_center, fov_width)
        total_gts_by_class = defaultdict(int)
        for s_token in self.sample_tokens:
            gts_for_token = self.gt_boxes_by_token[s_token]
            if fov_center is not None and fov_width is not None:
                 gts_for_token = [box for box in gts_for_token if is_box_in_fov(box, fov_center, fov_width)]
            for gt_box in gts_for_token:
                total_gts_by_class[get_general_class_name(gt_box.label)] += 1
        return all_class_results, total_gts_by_class

    def _calculate_final_map(self, all_class_results, total_gts_by_class):
        ap_by_class = {}
        all_class_names = total_gts_by_class.keys() | all_class_results.keys()
        for class_name in all_class_names:
            ap = self._calculate_ap(all_class_results[class_name], total_gts_by_class[class_name])
            ap_by_class[class_name] = ap
        valid_aps = [ap for ap in ap_by_class.values() if ap is not None]
        mAP = np.mean(valid_aps) if valid_aps else 0.0
        return mAP, ap_by_class

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Custom mAP evaluation for nuScenes detection results.')
    parser.add_argument('input_path', type=str, help='Path to the inference results JSON file or a directory containing result files.')
    parser.add_argument('--dataroot', type=str, default='/data2/yoshida/v1.0-mini-root', help='Path to the nuScenes data root.')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='nuScenes version.')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU threshold for matching (for iou metric).')
    parser.add_argument('--score-threshold', type=float, default=0.1, help='Confidence score threshold for predictions.')
    parser.add_argument('--plot-dir', type=str, default=None, help='Directory to save BEV visualization plots. Requires matplotlib.')
    parser.add_argument('--fov-center', type=float, default=None, help='Center of the FOV in degrees (0 is forward).')
    parser.add_argument('--fov-width', type=float, default=None, help='Width of the FOV in degrees.')
    parser.add_argument('--match-metric', type=str, default='dist', choices=['iou', 'dist'], help='Matching metric: iou or dist (center distance).')
    args = parser.parse_args()

    if args.plot_dir and not MATPLOTLIB_AVAILABLE: exit("[ERROR] Matplotlib not found, required for plotting.")
    try: from shapely.geometry import Polygon
    except ImportError: exit("[ERROR] Shapely library not found.")

    evaluator = CustommAPEvaluator(args.input_path, args.dataroot, args.version, args.plot_dir, args.score_threshold)
    evaluator.evaluate(args.iou_threshold, args.fov_center, args.fov_width, args.match_metric)