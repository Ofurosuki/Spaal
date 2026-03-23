#!/usr/bin/env python3
"""
Summarize KITTI mAP evaluation results from *_map_official.txt files.

Collects results organized by sync angle and model (PointPillars / PV-RCNN),
and displays them in a single table.

Usage:
    uv run python datasets_generator/summarize_eval_results.py
    uv run python datasets_generator/summarize_eval_results.py --base-dir /data2/yoshida --dir-src kitti_100/swin_denoised_64_bin
"""

import os
import re
import glob
import argparse


ANGLES = ['0_2', '0_8', '1', '2', '5', '11', '22', '45']
DIFFICULTIES = ['ALL', 'EASY', 'MODERATE', 'HARD']

MODELS = [
    ('PointPillars', 'predictions_pp_*_map_official.txt'),
    ('PV-RCNN',      'predictions_pvrcnn_*_converted_map_official.txt'),
]


def parse_map_official_txt(filepath):
    """
    Parse *_map_official.txt and return AP% values by difficulty.

    Expected lines (from calculate_map_kitti_official.py):
        ALL             0.1234          12.34
        EASY            0.2345          23.45
        MODERATE        0.3456          34.56
        HARD            0.4567          45.67
    """
    results = {}
    pattern = re.compile(r'^(ALL|EASY|MODERATE|HARD)\s+([\d.]+)\s+([\d.]+)', re.MULTILINE)
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        for m in pattern.finditer(content):
            difficulty = m.group(1)
            ap_pct = float(m.group(3))
            results[difficulty] = ap_pct
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
        description='Summarize KITTI mAP results by angle and model'
    )
    parser.add_argument('--base-dir', default='/data2/yoshida')
    parser.add_argument('--dir-src', default='kitti_100/swin_denoised_64_bin')
    parser.add_argument('--angles', nargs='+', default=ANGLES)
    args = parser.parse_args()

    src_dir = os.path.join(args.base_dir, args.dir_src)

    # --- Collect results ---
    data = {}  # data[angle][model_name] = {diff: ap_pct} or None
    for angle in args.angles:
        angle_dir = os.path.join(src_dir, angle)
        data[angle] = {}
        for model_name, pat in MODELS:
            filepath = find_latest(angle_dir, pat)
            if filepath is None:
                data[angle][model_name] = None
            else:
                parsed = parse_map_official_txt(filepath)
                data[angle][model_name] = parsed if parsed else None

    # --- Display ---
    col_angle = 8
    col_model = 14
    col_val   = 11

    header = (
        f"{'Angle':<{col_angle}} {'Model':<{col_model}}"
        + "".join(f"{d:>{col_val}}" for d in DIFFICULTIES)
    )
    sep = "-" * len(header)
    thick = "=" * len(header)

    print(thick)
    print("KITTI mAP Evaluation Summary")
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
                vals = "".join(f"{'N/A':>{col_val}}" for _ in DIFFICULTIES)
            else:
                vals = ""
                for diff in DIFFICULTIES:
                    if diff in results:
                        vals += f"{results[diff]:>{col_val - 1}.2f}%"
                    else:
                        vals += f"{'N/A':>{col_val}}"

            print(f"{angle_label:<{col_angle}} {model_name:<{col_model}}{vals}")

        print(sep)

    print()


if __name__ == '__main__':
    main()
