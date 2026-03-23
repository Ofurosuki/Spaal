#!/usr/bin/env python3
"""
Summarize nuScenes-style mAP evaluation results from *_map_nuscenes_style.txt files.

Collects results organized by sync angle and model (PointPillars / PV-RCNN),
and displays them in a single table.

Usage:
    uv run python datasets_generator/summarize_eval_results_nuscenes_style.py
    uv run python datasets_generator/summarize_eval_results_nuscenes_style.py \
        --base-dir /data2/yoshida --dir-src kitti_100/swin_denoised_64_bin
"""

import os
import re
import glob
import argparse


ANGLES = ['0_2', '0_8', '1', '2', '5', '11', '22', '45']
DIST_THS = [0.5, 1.0, 2.0, 4.0]

MODELS = [
    ('PointPillars', 'predictions_pp_*_map_nuscenes_style.txt'),
    ('PV-RCNN',      'predictions_pvrcnn_*_converted_map_nuscenes_style.txt'),
]


def parse_nuscenes_style_txt(filepath):
    """
    Parse *_map_nuscenes_style.txt and return AP% by distance threshold and mean AP.

    Expected lines (from calculate_map_nuscenes_style.py):
        0.5                       0.1234          12.34
        1.0                       0.2345          23.45
        2.0                       0.3456          34.56
        4.0                       0.4567          45.67

        Mean AP:                  0.3456          34.56
    """
    results = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Per-distance lines: "0.5   0.1234   12.34"
        dist_pattern = re.compile(r'^(\d+\.\d+)\s+([\d.]+)\s+([\d.]+)', re.MULTILINE)
        for m in dist_pattern.finditer(content):
            dist_th = float(m.group(1))
            ap_pct = float(m.group(3))
            results[dist_th] = ap_pct

        # Mean AP line: "Mean AP:   0.3456   34.56"
        mean_m = re.search(r'^Mean AP:\s+([\d.]+)\s+([\d.]+)', content, re.MULTILINE)
        if mean_m:
            results['mean'] = float(mean_m.group(2))

    except Exception as e:
        print(f"  Warning: failed to parse {filepath}: {e}")
    return results


def find_latest(directory, pattern):
    """Return the most recently modified file matching glob pattern, or None."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(
        description='Summarize nuScenes-style mAP results by angle and model'
    )
    parser.add_argument('--base-dir', default='/data2/yoshida')
    parser.add_argument('--dir-src', default='kitti_100/swin_denoised_64_bin')
    parser.add_argument('--angles', nargs='+', default=ANGLES)
    args = parser.parse_args()

    src_dir = os.path.join(args.base_dir, args.dir_src)

    # --- Collect results ---
    data = {}  # data[angle][model_name] = {dist_th: ap_pct, 'mean': ap_pct} or None
    for angle in args.angles:
        angle_dir = os.path.join(src_dir, angle)
        data[angle] = {}
        for model_name, pat in MODELS:
            filepath = find_latest(angle_dir, pat)
            if filepath is None:
                data[angle][model_name] = None
            else:
                parsed = parse_nuscenes_style_txt(filepath)
                data[angle][model_name] = parsed if parsed else None

    # --- Display ---
    col_angle = 8
    col_model = 14
    col_val   = 10

    col_headers = [f"AP@{d}m" for d in DIST_THS] + ['Mean AP']
    header = (
        f"{'Angle':<{col_angle}} {'Model':<{col_model}}"
        + "".join(f"{h:>{col_val}}" for h in col_headers)
    )
    sep = "-" * len(header)
    thick = "=" * len(header)

    print(thick)
    print("nuScenes-Style mAP Evaluation Summary (Center Distance Matching)")
    print(f"Source : {src_dir}")
    print(thick)
    print(header)
    print(sep)

    for angle in args.angles:
        first_row = True
        for model_name, _ in MODELS:
            angle_label = angle if first_row else ''
            first_row = False

            results = data[angle].get(model_name)
            if results is None:
                vals = "".join(f"{'N/A':>{col_val}}" for _ in col_headers)
            else:
                vals = ""
                for dist_th in DIST_THS:
                    if dist_th in results:
                        vals += f"{results[dist_th]:>{col_val - 1}.2f}%"
                    else:
                        vals += f"{'N/A':>{col_val}}"
                if 'mean' in results:
                    vals += f"{results['mean']:>{col_val - 1}.2f}%"
                else:
                    vals += f"{'N/A':>{col_val}}"

            print(f"{angle_label:<{col_angle}} {model_name:<{col_model}}{vals}")

        print(sep)

    print()


if __name__ == '__main__':
    main()
