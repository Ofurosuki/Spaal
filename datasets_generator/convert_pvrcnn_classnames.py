#!/usr/bin/env python3
"""
PV-RCNN の推論結果 JSON に含まれるクラス名を KITTI 用クラス名に変換する。

PV-RCNN (kitti-3d-3class) の出力クラス名の対応:
  "vehicle.construction" -> "Car"      (Car / Pedestrian が共にこの名前で出力される問題あり)
  "vehicle.truck"        -> "Cyclist"
"""

import json
import argparse
import sys
from pathlib import Path

# PV-RCNN 出力名 → KITTI クラス名
CLASS_NAME_MAP: dict[str, str] = {
    "vehicle.construction": "Car",
    "vehicle.truck": "Cyclist",
    "vehicle.car": "Pedestrian",
}


def convert_names_in_obj(obj, name_map: dict[str, str]):
    """JSON オブジェクトを再帰的にたどり、クラス名を置換する。"""
    if isinstance(obj, dict):
        # キーが対象クラス名の場合はキーごと置換
        new_dict = {}
        for k, v in obj.items():
            new_key = name_map.get(k, k)
            new_dict[new_key] = convert_names_in_obj(v, name_map)
        return new_dict
    elif isinstance(obj, list):
        return [convert_names_in_obj(item, name_map) for item in obj]
    elif isinstance(obj, str):
        return name_map.get(obj, obj)
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="PV-RCNN JSON クラス名を KITTI 用に変換する"
    )
    parser.add_argument("input_json", help="変換元 JSON ファイルパス")
    parser.add_argument("output_json", help="変換後 JSON ファイルパス")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r") as f:
        data = json.load(f)

    converted = convert_names_in_obj(data, CLASS_NAME_MAP)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"[INFO] Converted: {input_path} -> {output_path}")
    print(f"[INFO] Class name mapping applied: {CLASS_NAME_MAP}")


if __name__ == "__main__":
    main()
