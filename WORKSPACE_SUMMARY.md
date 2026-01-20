# SPAAL v2 Workspace Summary

**Last Updated:** 2025-11-14
**Project:** SPAAL v2 (Simulator of the Physical Attack Against LiDARs)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Core Components](#core-components)
4. [Dataset Generation Pipeline](#dataset-generation-pipeline)
5. [Denoising Methods](#denoising-methods)
6. [Evaluation Pipeline](#evaluation-pipeline)
7. [Key Scripts and Commands](#key-scripts-and-commands)
8. [Data Formats and File Types](#data-formats-and-file-types)
9. [Important Parameters and Thresholds](#important-parameters-and-thresholds)
10. [Coordinate Systems](#coordinate-systems)
11. [Known Issues and Solutions](#known-issues-and-solutions)
12. [Model Checkpoints](#model-checkpoints)

---

## Project Overview

SPAAL v2 is a LiDAR attack/defense simulator that:
1. **Simulates** LiDAR sensor behavior with waveform-level detail
2. **Attacks** LiDAR sensors with HFR (High-Frequency Radar) spoofing
3. **Reconstructs** clean point clouds from attacked signals using:
   - Baseline method: Template subtraction
   - Neural network: 3D U-Net with Axial Attention

### Key Innovation
Unlike traditional point cloud manipulation, SPAAL v2 models the **complete waveform** (time-intensity signal) for each LiDAR measurement, enabling realistic attack simulation and physics-based denoising.

---

## Directory Structure

```
/home/yoshida/Spaal/
├── spaal2/                          # Core simulator library
│   └── core/
│       ├── dummy_lidar/             # LiDAR models (VLP16, VLP32c, HDL64E)
│       ├── dummy_spoofer/           # Attack models (HFR, continuous pulse)
│       └── dummy_outdoor.py         # Environmental effects
│
├── datasets_generator/              # Dataset creation tools
│   ├── hist_matrix_generator.py     # Main dataset generator
│   ├── bl2_to_bin_converter.py      # Convert bl2 → bin for inference
│   ├── convert_all_subdirs.sh       # Batch conversion script
│   └── 1108_kitti_dataset_eval_test.sh  # Full pipeline example
│
├── baseline_denoising/              # Template subtraction baseline
│   ├── template_subtraction_denoiser.py
│   └── run_template_denoising_full_pipeline.sh
│
├── HFR_Denoise/                     # Neural network denoiser
│   ├── model/                       # 3D U-Net + Axial Attention
│   ├── dataset/                     # PyTorch dataset loaders
│   ├── src/                         # Training/inference scripts
│   └── run/                         # Saved model checkpoints
│
├── kitti_eval_package/              # Evaluation tools
│   ├── calculate_map_nuscenes_style.py
│   ├── calculate_map_nuscenes_detailed.py
│   ├── calculate_map_nuscenes_camera_coords.py
│   ├── kitti_label_parser.py
│   └── calculate_map_kitti_style.py
│
├── reconstruction/                  # Signal reconstruction algorithms
│   ├── hfr_frequency_identifier_fourier.py
│   ├── peak_interval_reconstructor.py
│   └── run_pipeline.py
│
├── evaluation/                      # Quality metrics
│   ├── evaluate_reconstruction.py
│   └── evaluate_denoised.sh
│
├── torch_interface/                 # PyTorch dataset interface
│   └── dataset.py
│
└── example/                         # Attack scenario examples
    ├── ahfr.py
    └── continuous_pulse_with_perturbation.py
```

---

## Core Components

### 1. LiDAR Models

#### **Supported Models**
- **VLP16**: 16 channels, Velodyne Puck
- **VLP32c**: 32 channels, Velodyne Ultra Puck
- **HDL64E**: 64 channels, Velodyne HDL-64E (KITTI dataset)

#### **Key Implementation Files**
- `spaal2/core/dummy_lidar/dummy_lidar_vlp16.py`
- `spaal2/core/dummy_lidar/dummy_lidar_vlp32_pcd.py`
- `spaal2/core/dummy_lidar/dummy_lidar_hdl64e.py`

#### **Critical Implementation: Intensity=0 Handling**
KITTI datasets have **39.41% of vehicle points with intensity=0.0**. Without special handling, these would be filtered out.

**Solution** (in all dummy_lidar files):
```python
if pcd_intensity > 0:
    pulse_amplitude = pcd_intensity * self.intensity_to_amplitude_ratio
else:
    pulse_amplitude = 0.05 * self.intensity_to_amplitude_ratio  # Default: 0.05 * 12.0 = 0.6
```

### 2. Spoofer Models

#### **Attack Types**
- `DummySpooferAdaptiveHFR`: Adaptive high-frequency attack
- `DummySpooferAdaptiveHFRWithPerturbation`: With timing perturbation
- `DummySpooferContinuousPulse`: Continuous pulse attack
- `DummySpooferOff`: No attack (baseline/ground truth)

### 3. Data Structures

#### **VeloPoint**
```python
class VeloPoint:
    intensity: float
    channel: int
    timestamp: PreciseDuration
    azimuth: float
    altitude: float
    distance: float
    x, y, z: float
```

#### **Hist-matrix Format**
3D array representing LiDAR waveforms:
- **Shape**: `(H, W, D)` = Altitude × Azimuth × Histogram
  - H: Number of channels (16/32/64)
  - W: Horizontal resolution (typically 1800 for 0.2°)
  - D: Time/range bins (typically 800)

---

## Dataset Generation Pipeline

### Main Script: `hist_matrix_generator.py`

**Purpose:** Generate training/testing datasets with simulated HFR attacks

#### **Key Arguments**

| Argument | Values | Description |
|----------|--------|-------------|
| `--lidar-type` | PCD_HDL64E, PCD_VLP32c, PCD_VLP16, VLP16 | LiDAR model |
| `--pcd-directory` | path | Input point cloud directory |
| `--num-frames` | int | Number of frames to generate |
| `--output-dir` | path | Output directory |
| `--spoofer-type` | adaptive_hfr_perturbation, continuous_pulse, off | Attack type |
| `--spoofer-angle` | float (deg) | Spoofer azimuth angle |
| `--spoofer-altitude` | float (deg) | Spoofer altitude angle |
| `--time-resolution-ns` | float | Time resolution (default: 1.0ns) |
| `--sync-angle` | float | Synchronization angle step (default: 0.2°) |
| `--horizontal-resolution-deg` | float | Internal horizontal resolution |
| `--output-horizontal-resolution-deg` | float | Output resolution (auto if omitted) |
| `--start-frame` | int | Starting frame for batch processing |

#### **Auto-detection of Resolution**
When `--output-horizontal-resolution-deg` is not specified:
- **HDL-64E**: 0.0818° (4400 samples/rotation)
- **VLP32c/VLP16**: 0.2° (1800 samples/rotation)

#### **Example: Generate KITTI Dataset**
```bash
uv run python datasets_generator/hist_matrix_generator.py \
  --lidar-type PCD_HDL64E \
  --pcd-directory /d/testing/velodyne \
  --num-frames 100 \
  --output-dir ./kitti_datasets_hdl64e \
  --spoofer-type adaptive_hfr_perturbation \
  --spoofer-angle 0 \
  --spoofer-altitude 0 \
  --time-resolution-ns 1.0 \
  --sync-angle 1.0 \
  --start-frame 0
```

#### **Output Files (Default: .bl2 format)**
For each frame (e.g., `frame_000000/`):
- `signal.bl2`: Attacked waveform data (H×W×D, blosc2 compressed)
- `answer_matrix.bl2`: Ground truth ToF offsets (H×W)
- `labels.bl2`: 3-class labels (H×W×D, uint8)
- `config.json`: Metadata (vertical_angles, fov, time_resolution_ns, etc.)
- `angles.bl2`: Actual azimuth angles per measurement (optional)
- `timestamps.bl2`: Timestamp per horizontal position (optional)

---

## Denoising Methods

### 1. Baseline: Template Subtraction

**Script:** `baseline_denoising/template_subtraction_denoiser.py`

#### **Principle**
1. Build templates: Average signals at specific timestamps (mod attack period)
2. Subtract template from each signal
3. Zero out signals below threshold

#### **Key Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-template-samples` | 3 | Minimum samples to build template |
| `--min-peak-threshold` | 0.01 | Minimum peak after subtraction |
| `--chunk-size` | 50 | Frames per batch (memory optimization) |

#### **Full Pipeline Script**
```bash
bash baseline_denoising/run_template_denoising_full_pipeline.sh
```

**Pipeline steps:**
1. Template subtraction denoising (bl2 → bl2)
2. Convert to .bin format
3. Run inference with MMDetection3D (Docker)
4. Evaluate with nuScenes-style metrics

### 2. Neural Network: HFR_Denoise

**Architecture:** 3D U-Net + Axial Self-Attention

#### **Network Overview**
- **Input:** `(B, H, W, D)` histogram signals
- **Output:** `(B, 3, H, W, D)` class logits
  - Class 0: Background
  - Class 1: True LiDAR return
  - Class 2: HFR attack

#### **Attention Mechanisms**

| Type | Axis | Sequence Length | Purpose |
|------|------|-----------------|---------|
| AxialSelfAttentionW | Azimuth | 1800 | Ring-pattern attacks |
| AxialSelfAttentionH | Height | 64 | Cross-channel attacks |
| AxialSelfAttentionD | Range/Time | 800 | Temporal pulse patterns |

**Attention Configuration:**
```python
use_axial_attn = "whd"  # All three axes (default)
use_axial_attn = "w"    # Azimuth only
use_axial_attn = "h"    # Height only
use_axial_attn = "d"    # Depth only
use_axial_attn = False  # No attention (baseline)
```

#### **Key Components**
- **Circular Padding:** Preserves 360° azimuthal periodicity
- **Depthwise-Separable Conv3D:** Parameter-efficient
- **GroupNorm:** Batch-size independent normalization
- **Skip Connections:** Multi-scale feature fusion

#### **Training Configuration**
- **Loss:** Weighted CrossEntropy [1.0, 3.0, 6.0] + Dice Loss
- **Optimizer:** AdamW (lr=3e-4, weight_decay=5e-2)
- **Scheduler:** CosineAnnealingLR (50 epochs)
- **Precision:** Mixed precision (float16 forward, float32 backward)

#### **Inference Optimization**
- `--split-h`: Split H dimension to reduce GPU memory
- `--fp16`: Use float16 precision
- `--chunk-size`: Process frames in batches

#### **Model Checkpoints** (in `/HFR_Denoise/run/`)

| Model File | Description | Parameters |
|------------|-------------|------------|
| `pretrain_general_data_dim32_attn.pt` | Full "whd" attention | ~6.6M |
| `1030_dn_dim32_lat_attn_deg_0_kitti.pt` | KITTI 64-line | ~6.6M |
| `ablation_dn_dim32_no_attn.pt` | No attention | ~3.5M |
| `ablation_dn_dim32_axial_w.pt` | W-axis only | ~4.5M |
| `ablation_dn_dim32_axial_h.pt` | H-axis only | ~4.5M |
| `ablation_dn_dim32_axial_d.pt` | D-axis only | ~4.5M |

---

## Evaluation Pipeline

### Coordinate Systems

**CRITICAL:** KITTI uses **two coordinate systems**:

#### **Camera Coordinates** (for labels)
- x-axis: Right
- y-axis: Down
- z-axis: Forward

#### **LiDAR Coordinates** (for point clouds)
- x-axis: Forward
- y-axis: Left
- z-axis: Up

**Transformation:**
```python
x_lidar = z_camera
y_lidar = -x_camera
z_lidar = -y_camera
```

### Evaluation Methods

#### 1. **KITTI-Style Evaluation** (IoU-based)
**Script:** `kitti_eval_package/calculate_map_kitti_style.py`

- **Metric:** AP@IoU threshold (typically 0.7 for Car)
- **Coordinate:** Camera coordinates
- **Matching:** 3D bounding box IoU

#### 2. **nuScenes-Style Evaluation** (Distance-based)
**Script:** `kitti_eval_package/calculate_map_nuscenes_style.py`

- **Metric:** mAP with center distance thresholds [0.5, 1.0, 2.0, 4.0]m
- **Coordinate:** LiDAR coordinates (parser transforms automatically)
- **Matching:** 2D BEV center distance

**Key Arguments:**
```bash
--filter-kitti-range    # Filter predictions to KITTI forward range
                        # x: [0, 70]m, y: [-40, 40]m
```

#### 3. **Detailed nuScenes Evaluation** (with PR curves)
**Script:** `kitti_eval_package/calculate_map_nuscenes_detailed.py`

**Outputs:**
- Precision-Recall curves
- Confidence distribution histograms
- Per-threshold metrics
- Saves visualization as PNG

**Example:**
```bash
uv run python kitti_eval_package/calculate_map_nuscenes_detailed.py \
  /data2/yoshida/kitti_100/kitti_no_spoofer_bin/kitti_predictions_20251108_112225.json \
  --gt-label-dir /data2/yoshida/label_kitti/training/label_2 \
  --output-dir ./evaluation_results \
  --filter-kitti-range
```

### Coordinate System Verification

**Verification Result (from `verify_coordinate_system.py`):**
- Predictions from MMDetection3D are in **LiDAR coordinates**
- Parser transformation is **correct**
- BEV distance is **invariant to coordinate rotation** (scalar value)

**Evidence:**
```
Sample 000007:
  GT (Camera):  x=-0.69, y=1.69, z=25.01
  GT (LiDAR):   x=25.01, y=0.69, z=-1.69
  Prediction:   x=25.42, y=0.74, z=-1.45

  BEV distance (if camera): 37.18m ❌
  BEV distance (if LiDAR):   0.41m ✅
```

---

## Key Scripts and Commands

### Dataset Generation

#### **HDL-64E (KITTI-compatible)**
```bash
./test_hdl64e_generator.sh
```

#### **Batch Generation**
```bash
./generate_dataset_batches.sh
```

### Conversion

#### **bl2 to bin**
```bash
uv run python datasets_generator/bl2_to_bin_converter.py \
  --input-dir ./denoised_datasets \
  --output-dir ./bin_output \
  --format kitti \
  --min-peak-amplitude 0.01
```

**Parameters:**
- `--format`: `kitti` (4 elements: x,y,z,intensity) or `nuscenes` (5 elements: +ring)
- `--min-peak-amplitude`: Minimum amplitude threshold (default: 0.01)
- `--amplitude-to-intensity-ratio`: Conversion ratio (default: 25.5)

### Baseline Denoising

```bash
bash baseline_denoising/run_template_denoising_full_pipeline.sh
```

**Configurable variables in script:**
- `INPUT_DIR`: Input bl2 dataset directory
- `MIN_TEMPLATE_SAMPLES`: Template building threshold (default: 3)
- `MIN_PEAK_THRESHOLD`: Peak filtering threshold (default: 0.01)
- `CHUNK_SIZE`: Batch size (default: 50)

### Neural Network Denoising

#### **Training**
```bash
cd HFR_Denoise
uv run python src/train.py \
  --root-path ./data \
  --hidden-dim 32 \
  --use-axial-attn whd \
  --epochs 50 \
  --batch-size 2
```

#### **Inference**
```bash
uv run python src/inference.py \
  --model-path ./run/best_model.pt \
  --input-dir ./datasets/test \
  --output-dir ./denoised_output \
  --split-h \
  --fp16
```

### Evaluation

#### **nuScenes-style (recommended)**
```bash
uv run python kitti_eval_package/calculate_map_nuscenes_style.py \
  predictions.json \
  --gt-label-dir /data2/yoshida/label_kitti/training/label_2 \
  --filter-kitti-range
```

#### **KITTI-style**
```bash
uv run python kitti_eval_package/calculate_map_kitti_style.py \
  predictions.json \
  --gt-label-dir /data2/yoshida/label_kitti/training/label_2
```

### Visualization

#### **Hist-matrix Visualizer**
```bash
uv run python datasets_generator/hist_matrix_visualizer.py \
  --npz-file ./datasets/sample.npz \
  --pcd-directory ./pcd_data \
  --frame 0
```

#### **Intensity Distribution Analysis**
```bash
uv run python analyze_nuscenes_intensity.py \
  --bin-dir /data2/yoshida/kitti_100/kitti_no_spoofer_bin \
  --num-samples 20 \
  --output-dir ./intensity_analysis
```

---

## Data Formats and File Types

### .bl2 Format (blosc2 compressed)

**Advantages:**
- **Fast I/O:** 10-100x faster than .npz
- **High compression:** ~50% smaller than .npz
- **Memory efficient:** Streaming decompression

**Loading:**
```python
import blosc2
with open('signal.bl2', 'rb') as f:
    packed = f.read()
signal = blosc2.unpack_array(packed)
```

### .bin Format (KITTI)

**KITTI format:** Float32 little-endian, 4 or 5 values per point
```python
points = np.fromfile('000000.bin', dtype=np.float32)
if points.shape[0] % 4 == 0:
    points = points.reshape(-1, 4)  # x, y, z, intensity
else:
    points = points.reshape(-1, 5)  # x, y, z, intensity, ring
```

### .pcd Format

**Point Cloud Data:**
- ASCII or binary format
- Contains xyz + intensity + optional ring/timestamp
- Used as input to dataset generator

### config.json

**Metadata for each frame:**
```json
{
  "initial_azimuth_offset": 0.0,
  "vertical_angles": [-15.0, -13.0, ..., +3.0],
  "fov": 360.0,
  "time_resolution_ns": 1.0,
  "horizontal_resolution_deg": 0.2,
  "channels": 32
}
```

---

## Important Parameters and Thresholds

### Signal Filtering

#### **Two-Stage Filtering**

1. **Denoising Stage** (`template_subtraction_denoiser.py`)
   - `--min-peak-threshold`: Default 0.01
   - Filters weak residual signals after template subtraction

2. **Conversion Stage** (`bl2_to_bin_converter.py`)
   - `--min-peak-amplitude`: Default 0.01
   - Filters weak peaks during peak detection

#### **Intensity to Amplitude Conversion**
```python
intensity_to_amplitude_ratio = 12.0  # KITTI intensity (0-1) → pulse amplitude
```

**Why needed:**
- KITTI intensity range: [0.0, 1.0]
- Simulator amplitude threshold: 0.01
- Without conversion, most points would be filtered
- Special handling for intensity=0 (use default 0.05)

### Intensity Statistics (KITTI Dataset)

**From analysis of 20 samples (1,072,615 points):**
- Range: [0.0093, 0.7200]
- Mean: 0.2435
- Median: 0.2582
- Std: 0.1335
- Zero intensity: 0.00% (after conversion with min_peak_amplitude=0.01)

**Percentiles:**
- 25th: 0.1514
- 50th: 0.2582
- 75th: 0.3263
- 95th: 0.4597

### Resolution Settings

#### **HDL-64E (KITTI)**
- Vertical: 64 channels
- Vertical angles: +3.26° to -23.64°
- Horizontal (internal): 0.1° (3600 samples)
- Horizontal (output): 0.0818° (4400 samples) *auto-detected*
- Range bins: 800

#### **VLP-32c (nuScenes)**
- Vertical: 32 channels
- Horizontal: 0.2° (1800 samples)
- Range bins: 800

### KITTI Evaluation Range

**Forward-facing region only:**
- X-range: [0.0, 70.0]m (forward)
- Y-range: [-40.0, 40.0]m (lateral)

**Note:** Only 1 out of 648 predictions was outside this range (x < 0)

---

## Coordinate Systems

### Transformation Matrix (Rough Approximation)

```python
def camera_to_lidar(x_cam, y_cam, z_cam):
    x_lidar = z_cam
    y_lidar = -x_cam
    z_lidar = -y_cam
    return x_lidar, y_lidar, z_lidar
```

### BEV Distance Calculation

**In LiDAR coordinates (x=forward, y=left):**
```python
bev_dist = np.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)
```

**In Camera coordinates (x=right, z=forward):**
```python
bev_dist = np.sqrt((x_pred - x_gt)**2 + (z_pred - z_gt)**2)
```

**Key Insight:** BEV distance is a **scalar value**, invariant to coordinate frame rotation. Only the plane changes (x-y vs x-z).

---

## Known Issues and Solutions

### 1. CRLF Line Endings

**Problem:** Windows line endings in shell scripts
```
baseline_denoising/run_template_denoising_full_pipeline.sh: 行 7: \r': コマンドが見つかりません
```

**Solution:**
```bash
sed -i 's/\r$//' script.sh
```

### 2. Tensor Dimension Mismatch (Neural Network)

**Problem:** Model expects 4D, receives 5D or 6D
```
Expected 4D input [B, H, W, D], got [B, C, H, W, D]
```

**Solution:** Remove extra `.unsqueeze(0)` calls

### 3. Intensity=0 Filtering

**Problem:** 39.41% of KITTI vehicle points have intensity=0.0
**Solution:** Use default amplitude 0.05 when intensity=0 (see Core Components)

### 4. LiDAR-Specific Parameters

**Problem:** VLP32c doesn't accept HDL-64E-specific parameters

**Solution:** Conditional parameter passing in `hist_matrix_generator.py`:
```python
init_params = {...}  # Common params

if self.lidar_type == "PCD_HDL64E":
    init_params['horizontal_resolution_deg'] = self.horizontal_resolution_deg
    init_params['output_horizontal_resolution_deg'] = self.output_horizontal_resolution_deg
```

### 5. Coordinate System Confusion

**Problem:** Distrust of parser transformation

**Solution:** Created verification scripts proving predictions are in LiDAR coords:
- `verify_coordinate_system.py`
- `compare_coordinate_systems.py`
- `calculate_map_nuscenes_camera_coords.py` (0% AP proves not in camera coords)

---

## Model Checkpoints

### Location
```
/home/yoshida/Spaal/HFR_Denoise/run/
```

### Available Models

| Filename | Purpose | Architecture | Size |
|----------|---------|--------------|------|
| `best_model.pt` | General-purpose checkpoint | U-Net + attn | ~3.5M |
| `pretrain_general_data_dim32_attn.pt` | Pretrained full attention | dim=32, whd | ~6.6M |
| `1030_dn_dim32_lat_attn_deg_0_kitti.pt` | KITTI 64-line optimized | dim=32, whd | ~6.6M |
| `ablation_dn_dim16.pt` | Narrow network ablation | dim=16 | ~1.7M |
| `ablation_dn_dim24.pt` | Medium network ablation | dim=24 | ~2.5M |
| `ablation_dn_dim32_no_attn.pt` | No attention baseline | dim=32 | ~3.5M |
| `ablation_dn_dim32_axial_w.pt` | W-axis attention only | dim=32, w | ~4.5M |
| `ablation_dn_dim32_axial_h.pt` | H-axis attention only | dim=32, h | ~4.5M |
| `ablation_dn_dim32_axial_d.pt` | D-axis attention only | dim=32, d | ~4.5M |

### Loading Models

```python
from HFR_Denoise.model.denoise_model import DenoiseModel

model = DenoiseModel(
    hidden_dim=32,
    use_axial_attn="whd"
)
model.load_state_dict(torch.load('path/to/checkpoint.pt'))
model.eval()
```

---

## Data Paths (yoshida's Environment)

### KITTI Dataset
```
Ground Truth Labels: /data2/yoshida/label_kitti/training/label_2/
Point Clouds (.bin): /data2/yoshida/kitti_100/kitti_no_spoofer_bin/
Predictions (JSON):  /data2/yoshida/kitti_100/kitti_no_spoofer_bin/kitti_predictions_*.json
Denoised (bl2):      /data2/yoshida/kitti_100/nuscenes_denoised_64/horizontal/1/
```

### Test Data
```
Velodyne PCD: /d/testing/velodyne/
```

---

## Quick Reference Commands

### Generate Dataset + Evaluate (Full Pipeline)
```bash
# 1. Generate attacked dataset
uv run python datasets_generator/hist_matrix_generator.py \
  --lidar-type PCD_HDL64E \
  --pcd-directory /d/testing/velodyne \
  --num-frames 100 \
  --output-dir ./datasets/attacked \
  --spoofer-type adaptive_hfr_perturbation

# 2. Denoise (baseline method)
bash baseline_denoising/run_template_denoising_full_pipeline.sh

# OR denoise (neural network)
uv run python HFR_Denoise/src/inference.py \
  --model-path HFR_Denoise/run/best_model.pt \
  --input-dir ./datasets/attacked \
  --output-dir ./datasets/denoised

# 3. Convert to bin
uv run python datasets_generator/bl2_to_bin_converter.py \
  --input-dir ./datasets/denoised \
  --output-dir ./bin_output \
  --format kitti

# 4. Run inference (MMDetection3D in Docker)
docker exec pointpillars_container python demo/my_inference.py \
  /docker/path/bin_output \
  configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
  checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.pth \
  --output-json /docker/path/predictions.json

# 5. Evaluate
uv run python kitti_eval_package/calculate_map_nuscenes_style.py \
  predictions.json \
  --gt-label-dir /data2/yoshida/label_kitti/training/label_2 \
  --filter-kitti-range
```

### Check GPU Memory
```bash
nvidia-smi
```

### Docker Commands
```bash
docker ps  # List running containers
docker exec -it pointpillars_container bash  # Enter container
docker inspect pointpillars_container  # Check mounts
```

---

## Dependencies (Key Packages)

From `pyproject.toml`:
- `numpy <2.0.0` - Core numerical operations
- `open3d ≥0.18.0` - Point cloud visualization
- `blosc2` - Fast compression/decompression
- `torch ≥2.0.0` - Neural network framework
- `matplotlib` - Visualization
- `tqdm` - Progress bars
- `simple-pcd-viewer` - Custom PCD viewer (from GitHub)

**Environment:** Uses `uv` for dependency management

---

## Tips and Best Practices

### Memory Optimization
1. Use `--chunk-size` for batch processing
2. Use `--split-h` for neural network inference on large LiDAR
3. Use `.bl2` format instead of `.npz` for datasets
4. Enable `--fp16` for inference if GPU supports it

### Dataset Generation
1. Always check auto-detected resolution matches your LiDAR model
2. Use `--start-frame` for resuming interrupted batch generation
3. Monitor disk space - 100 frames ≈ 20-50 GB (compressed)

### Evaluation
1. Always use `--filter-kitti-range` for KITTI evaluation
2. Use detailed evaluation for PR curves and debugging
3. Verify coordinate system with verification scripts if results are unexpected

### Debugging
1. Check intensity distribution if detection performance is poor
2. Visualize hist-matrices to verify attack simulation
3. Compare baseline vs neural network denoising on same data
4. Use ablation models to understand attention contribution

---

## Future Work / TODOs

1. **Dataset Expansion**
   - Generate more diverse attack scenarios (varying angles, amplitudes)
   - Include multi-spoofer attacks

2. **Model Improvements**
   - Experiment with transformer-based architectures
   - Test on real-world LiDAR data with physical attacks

3. **Evaluation Enhancement**
   - Add Waymo Open Dataset evaluation
   - Implement tracking metrics (MOTA, MOTP)

4. **Optimization**
   - TensorRT/ONNX export for faster inference
   - Quantization for edge deployment

---

**End of Document**
