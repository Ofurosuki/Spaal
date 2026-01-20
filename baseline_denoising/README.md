# Baseline Denoising: Template Subtraction

This directory contains a baseline denoising method that uses timestamp-based template subtraction to remove HFR (High Frequency Radar) attack signals from LiDAR data.

## Method Overview

The denoising algorithm works as follows:

1. **Template Construction**: Collect all signals with the same timestamp across all samples and compute their average. This average serves as a "template" representing the typical signal pattern for that timestamp.

2. **Template Subtraction**: For each signal, subtract the corresponding timestamp template. This removes common patterns (like synchronized HFR attacks) while preserving unique legitimate returns.

3. **Non-negative Clipping**: Clip the result to non-negative values to ensure valid signal data.

## Key Insight

Since the LiDAR uses synchronized timestamps (controlled by `--sync-angle` parameter), measurements taken at the same timestamp will experience similar HFR attack patterns. By averaging signals with the same timestamp, we can estimate and remove the attack component.

## Files

- `template_subtraction_denoiser.py`: Main denoising implementation
- `run_template_denoising.sh`: Batch processing script for multiple SYNC_ANGLE variants
- `README.md`: This file

## Usage

### Single Directory

```bash
uv run python baseline_denoising/template_subtraction_denoiser.py \
  --input-dir D:/eval_cvpr2026/data/attacked_bl2/sync_1 \
  --output-dir D:/eval_cvpr2026/data/denoised_bl2/sync_1 \
  --min-template-samples 10
```

### Batch Processing (All SYNC_ANGLE Variants)

```bash
./baseline_denoising/run_template_denoising.sh
```

## Parameters

- `--input-dir`: Directory containing sample subdirectories with `signal.bl2` and `timestamps.bl2`
- `--output-dir`: Directory to save denoised results
- `--min-template-samples`: Minimum number of samples required to create a template for a timestamp (default: 2)
- `--min-peak-threshold`: Minimum peak intensity for denoised signal. If max(denoised_signal) < threshold, set signal to 0 (default: -1.0, disabled)
- `--chunk-size`: Number of samples to process at once to reduce memory usage (default: 20)

## Input Requirements

Each sample subdirectory must contain:
- `signal.bl2`: 3D array (channels, horizontal_resolution, time_samples)
- `timestamps.bl2`: 2D array (channels, horizontal_resolution) with timestamp values in nanoseconds
- `config.json`: Configuration file (will be copied to output)

## Output

For each sample, the denoiser outputs:
- `signal.bl2`: Denoised signal data
- `config.json`: Copy of original configuration
- `labels.bl2`: Copy of original labels (if present)
- `answer_matrix.bl2`: Copy of ground truth (if present)
- `angles.bl2`: Copy of azimuth angles (if present)
- `timestamps.bl2`: Copy of timestamps

Additionally, a `denoising_stats.json` file is saved in the output directory with:
- Number of templates created
- Number of samples processed
- List of timestamps with templates

## Evaluation

After denoising, evaluate the results using:

```bash
./evaluation/evaluate_denoised.sh
```

This will compare the denoised data against ground truth using the evaluation pipeline.

## Algorithm Details

### Template Construction Phase (Chunked Processing)

To reduce memory usage, samples are processed in chunks (default: 20 samples per chunk):

For each chunk:
1. Collect all signals (across chunk samples) with each timestamp
2. Filter signals with non-zero energy (sum > 0.01)
3. Compute running sum and count for each timestamp
4. Update global statistics (sum, count) for each timestamp
5. Clear chunk data to free memory

After all chunks:
1. For each timestamp with at least `min_template_samples` signals:
   - Compute the element-wise mean: template = sum / count
   - Store as the template for that timestamp

This chunked approach allows processing large datasets (e.g., 81 samples) without loading all signals into memory at once.

### Denoising Phase

For each pixel in each sample:
1. Get the timestamp value for that pixel
2. If a template exists for that timestamp:
   - Subtract the template from the signal
   - Clip negative values to 0
   - If `min_peak_threshold` is set and max(denoised_signal) < threshold:
     - Set entire signal to 0 (treat as noise)
3. If no template exists:
   - Keep the original signal unchanged

## Limitations

- Requires multiple samples with synchronized timestamps
- May not work well if legitimate returns also have synchronized timestamps
- Templates need sufficient samples (`min_template_samples`) to be reliable
- Cannot remove attacks that vary significantly across samples with the same timestamp

## Expected Performance

This baseline method should effectively remove consistent HFR attacks that align with LiDAR synchronization. Performance depends on:
- Number of samples available for template construction
- Consistency of HFR attack patterns
- Diversity of legitimate returns at each timestamp
