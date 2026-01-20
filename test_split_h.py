#!/usr/bin/env python3
"""
Test split_h functionality
"""
import torch
import sys
sys.path.append('/home/yoshida/Spaal')

print("Testing split_h functionality")
print("=" * 80)

# Simulate input
H = 64
x = torch.randn(1, 1, H, 1800, 800)
print(f"Input shape: {x.shape} (B, C, H, W, D)")

# Test split_h logic
split_h = 2
chunk_size = (H + split_h - 1) // split_h
print(f"\nsplit_h = {split_h}")
print(f"chunk_size = {chunk_size}")

print(f"\nProcessing chunks:")
for i in range(0, H, chunk_size):
    end_i = min(i + chunk_size, H)
    x_chunk = x[:, :, i:end_i, :, :]
    print(f"  Chunk {i//chunk_size}: range [{i}:{end_i}], shape = {x_chunk.shape}")

# Check if this matches expected behavior
print("\n" + "=" * 80)
print("Expected behavior:")
print(f"  Split {H} into {split_h} chunks")
print(f"  Each chunk should be approximately {H//split_h} lines")
print(f"  Chunk shapes should be (1, 1, ~{H//split_h}, 1800, 800)")

# Test with actual model forward pass simulation
print("\n" + "=" * 80)
print("Testing with model simulation:")

try:
    # Load model
    from HFR_Denoise.pipeline.denoise_pipeline import DenoisePipeline

    # This will fail but let's see the exact error
    import os
    ckpt_path = "/home/yoshida/Spaal/path/to/checkpoint.pt"  # Placeholder

    print("Model would process each chunk with Conv3d expecting 5D input")
    print("If getting 6D, something is wrong in the forward_logits function")

except Exception as e:
    print(f"Cannot test with actual model: {e}")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("If error shows [1, 1, 1, 40, 1800, 808] (6 dimensions):")
print("  - Extra dimension is being added somewhere")
print("  - Check if model's Conv3d is receiving x_chunk correctly")
print("  - The chunk shape (1, 1, 32, 1800, 800) is correct (5D)")
print("  - Problem might be in model's internal operations")
