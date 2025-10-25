# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPAAL v2 (Simulator of the Physical Attack Against LiDARs) is a simulator that:
1. Models LiDAR laser pulses and their reflected waveforms for each scan
2. Simulates HFR (High Frequency Radars) attacks - jamming attacks using high-frequency external lasers (spoofing)
3. Provides reconstruction modules to recover signals from jammed data

The core concept: Light used in ToF (Time-of-Flight) distance calculation is modeled as 1D time-intensity data. For each LiDAR measurement, the simulator generates reflected waveforms and attack waveforms, then calculates ToF from the composite signal.

## Development Environment

This project uses `uv` for dependency management. All Python commands should be prefixed with `uv run`:

```bash
uv run python script.py
uv run example/ahfr.py
```

## Core Architecture

### Signal Processing Flow

1. **LiDAR Simulation** (`spaal2/core/dummy_lidar/`)
   - `DummyLidarVLP16`, `PcdLidarVLP32c`, `PcdLidarHDL64E`: Different LiDAR models
   - Each scan() call generates a `MeasurementConfig` and a time-intensity signal
   - The `receive()` method processes composite signals to extract points

2. **Spoofer Simulation** (`spaal2/core/dummy_spoofer/`)
   - `DummySpooferAdaptiveHFR`: Adaptive high-frequency attack
   - `DummySpooferAdaptiveHFRWithPerturbation`: With timing perturbation
   - `DummySpooferContinuousPulse`: Continuous pulse attack
   - `DummySpooferOff`: No attack (baseline)
   - Each spoofer generates attack signals that are combined with legitimate LiDAR returns

3. **Environment Effects** (`spaal2/core/`)
   - `DummyOutdoor`: Simulates outdoor reflections
   - `apply_noise()`: Adds noise to signals
   - `gen_sunlight()`: Generates sunlight interference

4. **Dataset Generation** (`datasets_generator/`)
   - `hist_matrix_generator.py`: Main script for generating training datasets
   - **Default output format: .bl2** (blosc2 compressed) for efficient storage and fast loading
   - Can also output .npz files containing hist-matrices (Altitude × Azimuth × Histogram data)
   - Supports batch processing for large-scale dataset generation

5. **Reconstruction** (`reconstruction/`)
   - `hfr_frequency_identifier_fourier.py`: Identifies HFR attack frequency using FFT
   - `peak_interval_reconstructor.py`: Reconstructs clean signals by removing attack peaks
   - `run_pipeline.py`: Full pipeline (identify → reconstruct → save)

6. **Evaluation** (`evaluation/`)
   - `evaluate_reconstruction.py`: Compares reconstructed vs ground truth using MAE/MSE

7. **PyTorch Interface** (`torch_interface/`)
   - `dataset.py`: PyTorch Dataset for loading hist-matrix data with blosc2 compression

### Key Data Structures

- **VeloPoint**: Represents a single LiDAR point with intensity, channel, timestamp, azimuth, altitude, distance, and xyz coordinates
- **MeasurementConfig**: Holds configuration for a single LiDAR measurement (timestamp, duration, angles)
- **PreciseDuration**: High-precision time representation in nanoseconds
- **Hist-matrix**: 3D array (altitude × azimuth × histogram_data) representing LiDAR waveforms

### Hist-matrix Format

Output .bl2 files (default) and .npz files contain:
- `signals`: The actual histogram data (waveforms)
- `answer_matrix`: Ground truth offsets (not peak indices)
- `initial_azimuth_offsets`: Horizontal offset angles for first scan point
- `vertical_angles`: List of vertical angles for each channel
- `fov`: Field of view (360° for full-rotation LiDARs)
- `time_resolution_ns`: Time resolution in nanoseconds
- `labels`: 3-class labels (HFR pulse, true pulse, other) as uint8

## Common Commands

### Generate Datasets

**HDL-64E dataset (KITTI-compatible)**:
```bash
./test_hdl64e_generator.sh

# Or manually:
uv run python datasets_generator/hist_matrix_generator.py \
  --lidar-type PCD_HDL64E \
  --pcd-directory D:/testing/velodyne \
  --num-frames 2 \
  --output-dir ./lidar_datasets_hdl64e \
  --time-resolution-ns 1.0 \
  --sync-angle 1.0 \
  --start-frame 0 \
  --output-horizontal-resolution-deg 0.2
# Outputs .bl2 files by default
```

**VLP32c dataset (nuScenes-compatible)**:
```bash
uv run python datasets_generator/hist_matrix_generator.py \
  --lidar-type PCD_VLP32c \
  --pcd-directory ./nuscenes_data \
  --num-frames 1 \
  --output-dir ./pcd_datasets \
  --spoofer-angle 90 \
  --spoofer-altitude 50
# Outputs .bl2 files by default
```

**Batch dataset generation**:
```bash
./generate_dataset_batches.sh
# Generates large datasets in batches to manage memory
```

### Visualize Hist-matrix

```bash
uv run python datasets_generator/hist_matrix_visualizer.py \
  --npz-file ./pcd_datasets/lidar_signal.npz \
  --pcd-directory ./nuscenes_data \
  --frame 0
```

### Run Attack Example

```bash
uv run example/ahfr.py
uv run example/ahfr_with_perturbation.py
uv run example/continuous_pulse_with_perturbation.py
```

### Reconstruction Pipeline

```bash
uv run python reconstruction/run_pipeline.py \
  input.npz \
  output.npz \
  --id-threshold 0.1 \
  --id-peak-count 3 \
  --recon-tolerance 1.5 \
  --recon-min-run 3
```

### Evaluation

```bash
uv run python evaluation/evaluate_reconstruction.py \
  --reconstructed-npz ./output.npz \
  --ground-truth-npz ./input.npz \
  --evaluation-method mae
```

## LiDAR Model Specifications

### Resolution Auto-detection

When `--output-horizontal-resolution-deg` is not specified, it auto-detects:
- **HDL-64E**: 0.0818° (4400 samples per rotation)
- **VLP32c/VLP16**: 0.2° (1800 samples per rotation)

### Channel Configuration

- **HDL-64E**: 64 channels (vertical angles: +3.26° to -23.64°)
- **VLP32c**: 32 channels
- **VLP16**: 16 channels

### Critical Implementation Details

#### 1. Intensity=0 Handling (dummy_lidar_hdl64e.py, dummy_lidar_vlp32_pcd.py)

KITTI datasets have 39.41% of vehicle points with intensity=0.0. Without special handling, these would be filtered out (amplitude < 0.01 threshold).

**Solution**: When intensity=0, use default value of 0.05:
```python
if pcd_intensity > 0:
    pulse_amplitude = pcd_intensity * self.intensity_to_amplitude_ratio
else:
    pulse_amplitude = 0.05 * self.intensity_to_amplitude_ratio  # Default: 0.05 * 12.0 = 0.6
```

#### 2. LiDAR-Specific Parameters (hist_matrix_generator.py)

**Problem**: VLP32c does not accept HDL-64E-specific parameters (`horizontal_resolution_deg`, `output_horizontal_resolution_deg`).

**Solution**: Conditional parameter passing:
```python
init_params = {
    'pcd_file_path': None,
    'lidar_position': np.array([0.0, 0.0, 0.0]),
    'amplitude': lidar_amplitude_range[0],
    'pulse_width': PreciseDuration(nanoseconds=lidar_pulse_width_ns),
    # ... common params
}

# HDL64E-only parameters
if self.lidar_type == "PCD_HDL64E":
    init_params['horizontal_resolution_deg'] = self.horizontal_resolution_deg
    init_params['output_horizontal_resolution_deg'] = self.output_horizontal_resolution_deg
```

#### 3. Intensity to Amplitude Conversion

```python
intensity_to_amplitude_ratio = 12.0  # KITTI intensity (0-1) → pulse amplitude
```

Amplitude filtering threshold in `get_peak_time_and_amplitude()` is 0.01, so intensity=0 points need the default value workaround.

## Dataset Arguments Reference

Key arguments for `hist_matrix_generator.py`:
- `--lidar-type`: PCD_HDL64E | PCD_VLP32c | PCD_VLP16 | VLP16
- `--pcd-directory`: Directory containing .pcd or .bin files
- `--num-frames`: Number of frames to generate
- `--output-dir`: Output directory for dataset files (default format: .bl2)
- `--spoofer-angle`: Spoofer azimuth angle (degrees)
- `--spoofer-altitude`: Spoofer altitude angle (degrees)
- `--spoofer-type`: adaptive_hfr_perturbation | continuous_pulse | off
- `--time-resolution-ns`: Time resolution (default: 1.0ns)
- `--sync-angle`: Synchronization angle step (default: 0.2°)
- `--horizontal-resolution-deg`: Internal horizontal resolution (default: 0.1° for HDL64E)
- `--output-horizontal-resolution-deg`: Output resolution (auto-detected if omitted)
- `--start-frame`: Starting frame index for batch processing

## Directory Structure

```
spaal2/
  core/               # Core simulator components
    dummy_lidar/      # LiDAR models (VLP16, VLP32c, HDL64E)
    dummy_spoofer/    # Attack models (HFR, continuous pulse, etc.)
datasets_generator/   # Hist-matrix dataset generation
reconstruction/       # Signal reconstruction algorithms
evaluation/           # Reconstruction quality metrics
torch_interface/      # PyTorch dataset loader
example/              # Example attack scenarios
```

## Dependencies

Key dependencies (see pyproject.toml):
- numpy <2.0.0 (compatibility with existing code)
- open3d ≥0.18.0 (point cloud visualization)
- blosc2 (fast compression for large datasets)
- simple-pcd-viewer (from GitHub, custom PCD viewer)

## File Formats

- **.pcd**: Point Cloud Data files (ASCII or binary)
- **.bin**: KITTI binary point cloud format (x,y,z,intensity as float32)
- **.bl2**: Blosc2 compressed arrays (default output format for datasets) - optimized for fast I/O and efficient storage
- **.npz**: Compressed NumPy arrays (hist-matrices, labels, metadata) - alternative format for compatibility