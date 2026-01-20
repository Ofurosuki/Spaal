#!/usr/bin/env python3
"""
Test script to diagnose and fix denoise pipeline input dimensions.
"""
import sys
import os
import torch
import numpy as np
import blosc2
import glob

# Test 1: Load a sample signal.bl2 file and check its shape
def test_load_signal():
    print("=" * 80)
    print("Test 1: Loading signal.bl2 file")
    print("=" * 80)

    # Find a sample signal file (from 1111_test.sh)
    input_dir = "/data2/yoshida/kitti_100/attacked_64/1"
    sample_paths = glob.glob(f"{input_dir}/*/signal.bl2")

    if not sample_paths:
        print(f"No signal.bl2 files found in {input_dir}!")
        print("Trying alternative paths...")
        sample_paths = glob.glob("/data2/yoshida/kitti_100/**/signal.bl2", recursive=True)
        if not sample_paths:
            print("No signal.bl2 files found anywhere!")
            return None

    sample_path = sample_paths[0]
    print(f"Loading: {sample_path}")

    with open(sample_path, 'rb') as f:
        packed_signal = f.read()

    signal_data = blosc2.unpack_array(packed_signal)
    print(f"Loaded signal shape: {signal_data.shape}")
    print(f"Data type: {signal_data.dtype}")
    print(f"Min value: {signal_data.min():.4f}")
    print(f"Max value: {signal_data.max():.4f}")

    return signal_data


# Test 2: Test different unsqueeze combinations
def test_unsqueeze_options(signal_data):
    print("\n" + "=" * 80)
    print("Test 2: Testing unsqueeze operations")
    print("=" * 80)

    original_shape = signal_data.shape
    print(f"Original numpy shape: {original_shape}")

    # Convert to torch
    x = torch.from_numpy(signal_data).float()
    print(f"After torch.from_numpy(): {x.shape}")

    # Option 1: One unsqueeze
    x1 = x.unsqueeze(0)
    print(f"After .unsqueeze(0): {x1.shape}")

    # Option 2: Two unsqueezes
    x2 = x.unsqueeze(0).unsqueeze(0)
    print(f"After .unsqueeze(0).unsqueeze(0): {x2.shape}")

    # Option 3: Check if signal already has batch/channel dims
    if len(signal_data.shape) == 3:
        print("\nSignal is 3D (H, W, D)")
        x_correct = x.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W, D)
        print(f"Correct shape for model: {x_correct.shape} (B, C, H, W, D)")
        return x_correct
    elif len(signal_data.shape) == 4:
        print("\nSignal is 4D (C, H, W, D) or (1, H, W, D)")
        x_correct = x.unsqueeze(0)  # -> (1, C, H, W, D) or (1, 1, H, W, D)
        print(f"Correct shape for model: {x_correct.shape} (B, C, H, W, D)")
        return x_correct
    elif len(signal_data.shape) == 5:
        print("\nSignal is already 5D (B, C, H, W, D)")
        print(f"No unsqueeze needed: {x.shape}")
        return x
    else:
        print(f"\nUnexpected shape dimension: {len(signal_data.shape)}")
        return None


# Test 3: Test with actual model architecture expectations
def test_model_input():
    print("\n" + "=" * 80)
    print("Test 3: Model input requirements")
    print("=" * 80)

    print("DenoiseModel expects:")
    print("  - Input shape: (B, C, H, W, D)")
    print("  - B: batch size (typically 1)")
    print("  - C: channels (typically 1 for signal)")
    print("  - H: height (number of LiDAR lines, e.g., 32 or 64)")
    print("  - W: width (horizontal steps, e.g., 1800)")
    print("  - D: depth (time samples, e.g., 800)")
    print()
    print("Conv3d layers expect 5D input: (B, C, H, W, D)")


# Test 4: Generate fix
def generate_fix(signal_data):
    print("\n" + "=" * 80)
    print("Test 4: Generating fix")
    print("=" * 80)

    ndim = len(signal_data.shape)

    if ndim == 3:
        print("Signal is 3D (H, W, D)")
        print("Fix: x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims")
        print("Result: (H, W, D) -> (1, 1, H, W, D)")

        # Test it
        x = torch.from_numpy(signal_data).float()
        x_fixed = x.unsqueeze(0).unsqueeze(0)
        print(f"Test: {signal_data.shape} -> {x_fixed.shape}")
        return ".unsqueeze(0).unsqueeze(0)"

    elif ndim == 4:
        print("Signal is 4D")
        first_dim = signal_data.shape[0]

        if first_dim == 1:
            print("First dimension is 1, likely (1, H, W, D)")
            print("Fix: x.unsqueeze(0)  # Add batch dim only")
            print("Result: (1, H, W, D) -> (1, 1, H, W, D)")

            x = torch.from_numpy(signal_data).float()
            x_fixed = x.unsqueeze(0)
            print(f"Test: {signal_data.shape} -> {x_fixed.shape}")
            return ".unsqueeze(0)"
        else:
            print(f"First dimension is {first_dim}, likely (C, H, W, D)")
            print("Fix: x.unsqueeze(0)  # Add batch dim only")
            print("Result: (C, H, W, D) -> (1, C, H, W, D)")

            x = torch.from_numpy(signal_data).float()
            x_fixed = x.unsqueeze(0)
            print(f"Test: {signal_data.shape} -> {x_fixed.shape}")
            return ".unsqueeze(0)"

    elif ndim == 5:
        print("Signal is already 5D (B, C, H, W, D)")
        print("Fix: No unsqueeze needed")
        return ""

    else:
        print(f"Unexpected dimension: {ndim}")
        return None


if __name__ == "__main__":
    print("Denoise Pipeline Input Dimension Diagnostic Tool")
    print()

    # Run tests
    signal_data = test_load_signal()

    if signal_data is not None:
        test_unsqueeze_options(signal_data)
        test_model_input()
        fix = generate_fix(signal_data)

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Signal shape: {signal_data.shape}")
        print(f"Recommended fix: {fix}")
        print()
        print("Update denoise_pipeline.py:")
        print(f"  x = torch.from_numpy(x_np).float(){fix}")
