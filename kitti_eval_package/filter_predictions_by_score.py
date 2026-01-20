"""
Filter prediction JSON by score threshold to reduce false positives.

This improves Precision and consequently mAP in nuScenes-style evaluation.
"""

import argparse
import json


def filter_predictions_by_score(input_json: str, output_json: str, score_threshold: float):
    """
    Filter predictions by score threshold.

    Args:
        input_json: Input prediction JSON file
        output_json: Output filtered JSON file
        score_threshold: Minimum score to keep
    """
    # Load predictions
    with open(input_json, 'r') as f:
        data = json.load(f)

    total_before = 0
    total_after = 0

    # Filter results
    if 'results' in data:
        for sample_token in data['results']:
            predictions = data['results'][sample_token]
            total_before += len(predictions)

            # Filter by score
            filtered_predictions = [
                pred for pred in predictions
                if pred.get('detection_score', 0.0) >= score_threshold
            ]

            data['results'][sample_token] = filtered_predictions
            total_after += len(filtered_predictions)

    # Save filtered predictions
    with open(output_json, 'w') as f:
        json.dump(data, f)

    print(f"Filtered predictions:")
    print(f"  Input: {input_json}")
    print(f"  Output: {output_json}")
    print(f"  Score threshold: {score_threshold}")
    print(f"  Before: {total_before} predictions")
    print(f"  After: {total_after} predictions")
    print(f"  Kept: {total_after/total_before*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Filter predictions by score threshold'
    )
    parser.add_argument('input_json', type=str,
                        help='Input prediction JSON file')
    parser.add_argument('output_json', type=str,
                        help='Output filtered JSON file')
    parser.add_argument('--score-threshold', type=float, default=0.3,
                        help='Minimum score to keep (default: 0.3)')

    args = parser.parse_args()

    filter_predictions_by_score(
        args.input_json,
        args.output_json,
        args.score_threshold
    )
