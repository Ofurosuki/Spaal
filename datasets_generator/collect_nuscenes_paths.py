
import argparse
import json
import os
import random
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train as train_scene_names

def collect_paths(dataroot: str, version: str, num_samples: int, output_json: str):
    """
    Collects random .pcd.bin file paths from the nuScenes dataset, verifies
    their existence, and saves them to a JSON file along with their sample tokens.

    Args:
        dataroot: Path to the nuScenes root directory.
        version: NuScenes version (e.g., 'v1.0-trainval').
        num_samples: The number of samples to collect.
        output_json: Path to the output JSON file.
    """
    print(f"Initializing NuScenes SDK for version {version}...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    print("Filtering for scenes in the 'train' split...")
    # Get scene objects from the list of train scene names
    train_scenes = [s for s in nusc.scene if s['name'] in train_scene_names]
    
    all_samples = []
    for scene in tqdm(train_scenes, desc="Collecting samples from scenes"):
        current_sample_token = scene['first_sample_token']
        while current_sample_token:
            sample = nusc.get('sample', current_sample_token)
            all_samples.append(sample)
            current_sample_token = sample['next']

    print(f"Found {len(all_samples)} total samples in the train split. Shuffling...")
    random.shuffle(all_samples)

    collected_data = []
    print(f"Searching for {num_samples} existing LIDAR_TOP data files...")
    
    with tqdm(total=num_samples, desc="Verifying files") as pbar:
        for sample in all_samples:
            if len(collected_data) >= num_samples:
                break

            lidar_top_token = sample['data'].get('LIDAR_TOP')
            if not lidar_top_token:
                continue

            # The SDK provides the correct path to the .bin file directly
            file_path = nusc.get_sample_data_path(lidar_top_token)
            
            if os.path.exists(file_path) and file_path.endswith('.pcd.bin'):
                collected_data.append({
                    "token": lidar_top_token,
                    "path": file_path
                })
                pbar.update(1)

    if len(collected_data) < num_samples:
        print(f"\nWarning: Could only find {len(collected_data)} of the requested {num_samples} samples.")

    print(f"\nSaving {len(collected_data)} paths to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(collected_data, f, indent=2)

    print(f"Successfully created {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect and verify nuScenes LIDAR_TOP .pcd.bin file paths."
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        required=True,
        help="Path to the root directory of the nuScenes dataset (e.g., /data/sets/nuscenes)."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        help="NuScenes version (e.g., 'v1.0-trainval', 'v1.0-test'). Must match the dataroot."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random samples to collect."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="nuscenes_info.json",
        help="Path for the output JSON file."
    )
    args = parser.parse_args()

    collect_paths(
        dataroot=args.dataroot,
        version=args.version,
        num_samples=args.num_samples,
        output_json=args.output_json
    )
