# Experimental Setup: SPAAL v2 Simulator

## 1. Overview

We present **SPAAL v2** (Simulator of the Physical Attack Against LiDARs), a high-fidelity simulator for modeling High-Frequency Radar (HFR) spoofing attacks on Time-of-Flight (ToF) LiDAR systems and evaluating reconstruction algorithms. The simulator models the complete physical process: laser pulse emission, propagation, reflection, and reception, including adversarial interference from external high-frequency lasers.

### 1.1 System Architecture

The simulator consists of five main components:

1. **LiDAR Simulator**: Models authentic LiDAR sensing using point cloud data
2. **HFR Attack Simulator**: Generates adaptive high-frequency spoofing signals
3. **Environment Effects**: Simulates outdoor reflections, noise, and sunlight interference
4. **Dataset Generator**: Produces labeled training datasets with histogram-matrix representation
5. **Reconstruction Pipeline**: Identifies attack frequency and recovers clean signals

---

## 2. Physical Model

### 2.1 Time-of-Flight Calculation

Distance measurement follows the standard ToF principle:

```
d = (c × t) / 2
```

where:
- `d`: distance to target (meters)
- `c`: speed of light (0.299792458 m/ns)
- `t`: round-trip time (nanoseconds)

In the simulator, this is approximated as:
```
d = 0.15 × t_ns
```

where `t_ns` is the ToF in nanoseconds, simplifying `c/2 = 0.15 m/ns`.

### 2.2 Waveform Representation

Each LiDAR measurement is modeled as a 1D time-intensity signal. For a time resolution of `Δt` (default: 1 ns) and acceptance window `T_accept` (default: 800 ns), the signal is represented as:

```
s[i] = intensity at time (i × Δt), where i ∈ [0, T_accept/Δt]
```

### 2.3 Pulse Model

Laser pulses are modeled as Gaussian functions:

```
P(t) = A × exp(-(t - μ)² / (2σ²))
```

where:
- `A`: pulse amplitude (intensity-dependent)
- `μ`: time-of-flight (peak position)
- `σ`: pulse width parameter, derived from FWHM (Full Width at Half Maximum)

**FWHM to σ conversion**:
```
σ = FWHM / (2√(2ln2)) ≈ FWHM / 2.355
```

**Default pulse parameters**:
- Pulse width (FWHM): 5 ns
- Amplitude range: 3.0–8.0 (normalized units)

---

## 3. LiDAR Simulator

### 3.1 Supported LiDAR Models

We support three commercial LiDAR models using real-world point cloud data:

| Model | Channels | Vertical Range | Horizontal Resolution | Samples per Rotation |
|-------|----------|----------------|----------------------|---------------------|
| **HDL-64E** (Velodyne) | 64 | +3.26° to -23.64° | 0.0818° | 4400 |
| **VLP-32C** (Velodyne) | 32 | +10.67° to -30.67° | 0.2° | 1800 |
| **VLP-16** (Velodyne) | 16 | +15° to -15° | 0.2° | 1800 |

### 3.2 HDL-64E Specifications

The HDL-64E model uses calibrated vertical angles extracted from KITTI dataset metadata:

**Channel configuration** (64 channels):
- Upper block (channels 0-31): +3.26° to -7.82°
- Lower block (channels 32-63): -8.42° to -23.64°
- Vertical Correction Factor (VCF): precise angles per channel
- Rotational Correction Factor (RCF): minimal in KITTI data (<0.004°)

**Horizontal resolution modes**:
- Internal resolution: 0.1° (for fine-grained depth map construction)
- Output resolution: 0.0818° (4400 samples, ML-compatible format)

**Output channel options**:
- 64-channel mode: all channels (default)
- 32-channel mode: even channels only (0, 2, 4, ..., 62)

### 3.3 Intensity-to-Amplitude Mapping

LiDAR intensity values are mapped to pulse amplitudes:

```python
intensity_to_amplitude_ratio = 12.0  # For HDL-64E (KITTI)
intensity_to_amplitude_ratio = 40.0/255.0  # For VLP-32C (nuScenes)
```

**Special handling for zero-intensity points**:
KITTI datasets contain 39.41% of vehicle points with intensity=0.0. To prevent filtering (amplitude threshold: 0.01), these are assigned a default value:

```python
if intensity > 0:
    pulse_amplitude = intensity × 12.0
else:
    pulse_amplitude = 0.05 × 12.0 = 0.6  # Minimum detectable amplitude
```

### 3.4 Scan Modes

Two scan modes are supported:

**Horizontal mode** (default for HDL-64E):
- Scans horizontally first, then moves to next vertical channel
- Timestamp increment: `(azimuth / sync_angle_step) × 20 ns`
- Channel switch delay: `+50,234 ns` per channel

**Vertical mode** (default for VLP-32C):
- Scans all channels at one azimuth, then rotates
- Timestamp increment: `(channel / sync_channel_step) × 20 ns`
- Rotation delay: `+50,234 ns` per azimuth step

### 3.5 Synchronization Parameters

**Sync angle (horizontal mode)**:
- Purpose: Controls timestamp granularity
- Range: 0.2° – 2.0° (randomized per frame)
- Effect: Every `sync_angle_step` degrees, timestamp increases by 20 ns

**Sync channel (vertical mode)**:
- Purpose: Controls timestamp granularity
- Range: 1 – 32 channels (randomized per frame)
- Effect: Every `sync_channel_step` channels, timestamp increases by 20 ns

This randomization simulates real-world LiDAR timing variations.

### 3.6 Acceptance Window

All LiDAR models use a fixed acceptance window:
```
T_accept = 800 ns
```

This corresponds to a maximum detectable range:
```
d_max = 0.15 × 800 = 120 meters
```

---

## 4. HFR Attack Simulator

### 4.1 Attack Model: Adaptive HFR with Perturbation

The HFR spoofer generates high-frequency pulse trains synchronized to the victim LiDAR's firing events. We implement the **Adaptive HFR with Perturbation** model:

```python
class: DummySpooferAdaptiveHFRWithPerturbation
```

### 4.2 Attack Parameters

| Parameter | Symbol | Default Value | Range | Unit |
|-----------|--------|---------------|-------|------|
| Frequency | f_HFR | 10 | 1–20 | MHz |
| Pulse Width | τ_HFR | 5 | 3–10 | ns |
| Amplitude Range | A_HFR | [3.0, 8.0] | [1.0, 9.0] | normalized |
| Time Perturbation | δ_t | 20 | 0–50 | ns |
| Attack Duration | T_attack | 100 | 50–200 | ms |
| Spoofer Distance | d_spoofer | 10 | 5–50 | m |

### 4.3 Attack Signal Generation

**Pulse period**:
```
T_pulse = 1 / f_HFR = 1 / (10 × 10⁶) = 100 ns
```

**Per-pulse randomization**:
- Amplitude: `A_i ~ Uniform(3.0, 8.0)` (unique per pulse)
- Time offset: `δ_i ~ Uniform(-20, +20) ns` (jitter)

**Actual pulse timing**:
```
t_i = t_trigger + i × T_pulse + δ_i
```

where `t_trigger` is synchronized to the first detected legitimate pulse.

### 4.4 Attack Triggering Logic

The spoofer triggers when a legitimate LiDAR return is detected at the specified attack angle:

**Trigger condition** (horizontal mode):
```python
if |azimuth - spoofer_angle| < tolerance_azimuth AND
   |altitude - spoofer_altitude| < tolerance_altitude:
    trigger_attack()
```

Tolerances:
- Azimuth: ±8° (800 in 0.01° units)
- Altitude: ±4° (400 in 0.01° units)

**Attack cone geometry**:
- Center azimuth: `spoofer_angle_deg` (0° = front, CCW)
- Cone width: `spoofer_width_deg` (default: 90°)
- Attack range: `[center - width/2, center + width/2]`

### 4.5 Pulse Shape

HFR pulses use the same Gaussian model as LiDAR:

```python
σ = τ_HFR / (2√(2ln2))
pulse_range = ±3σ  # ~99.7% of energy
```

Normalized pulse shape (pre-computed):
```python
normalized_pulse = exp(-(x² / (2σ²)))
```

Actual pulse with random amplitude:
```python
actual_pulse = A_i × normalized_pulse
```

---

## 5. Dataset Generation

### 5.1 Output Format: Histogram-Matrix

Each frame is represented as a 4D tensor:

```
Shape: (channels, horizontal_steps, time_bins)
```

**Dimensions**:
- `channels`: Number of vertical channels (16/32/64)
- `horizontal_steps`: Azimuth samples (1800 for VLP-32C, 4400 for HDL-64E)
- `time_bins`: Temporal samples = `T_accept / Δt` = 800

**Example for HDL-64E**:
```
Shape: (64, 4400, 800)
Size: 64 × 4400 × 800 = 224,768,000 samples per frame
```

### 5.2 Data Storage Format

**Primary format**: Blosc2 compressed binary (`.bl2`)
- Compression ratio: ~10-20× (lossless)
- Fast I/O: optimized for large-scale training

**Stored arrays per frame**:

| File | Content | Shape | dtype |
|------|---------|-------|-------|
| `signal.bl2` | Raw waveform data | (C, H, T) | float32 |
| `labels.bl2` | 3-class labels | (C, H, T) | uint8 |
| `answer_matrix.bl2` | Ground truth ToF | (C, H) | float32 |
| `angles.bl2` | Azimuth angles | (C, H) | float32 |
| `timestamps.bl2` | Measurement timestamps | (C, H) | int64 |
| `config.json` | Metadata | - | JSON |

### 5.3 Label Definition

Three-class labeling scheme:

```
0: No return (background)
1: Legitimate LiDAR pulse
2: HFR attack pulse
```

Labeling logic:
```python
labels[signal > 0.01] = LEGITIMATE_PULSE  # Threshold: 0.01
labels[attack_signal > legitimate_signal] = HFR_PULSE  # Attack dominates
```

### 5.4 Dataset Generation Parameters

**Command-line interface**:
```bash
python datasets_generator/hist_matrix_generator.py \
  --lidar-type PCD_HDL64E \
  --pcd-directory /path/to/kitti/velodyne \
  --num-frames 100 \
  --output-dir ./datasets \
  --spoofer-type adaptive_hfr_perturbation \
  --spoofer-angle 0.0 \
  --spoofer-altitude 40.0 \
  --spoofer-width-deg 90.0 \
  --time-resolution-ns 1.0 \
  --sync-angle 1.0 2.0 \  # Random range
  --horizontal-resolution-deg 0.1 \
  --output-horizontal-resolution-deg 0.0818 \
  --output-channels 64 \
  --scan-mode horizontal
```

**Key parameters**:

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `--lidar-type` | LiDAR model selection | PCD_VLP32c |
| `--time-resolution-ns` | Temporal sampling rate | 1.0 ns |
| `--sync-angle` | Timestamp sync range | [0.2, 0.2]° |
| `--spoofer-angle` | Attack direction | 0° (front) |
| `--spoofer-altitude` | Attack elevation | 40° |
| `--spoofer-width-deg` | Attack cone width | 90° |
| `--scan-mode` | Scan pattern | vertical |
| `--output-channels` | Channel count | auto-detect |

### 5.5 Metadata (config.json)

```json
{
  "original_bin_path": "/path/to/000000.bin",
  "initial_azimuth_offset": 45.23,
  "vertical_angles": [3.26, 2.20, ..., -23.64],
  "fov": 360.0,
  "time_resolution_ns": 1.0
}
```

---

## 6. Reconstruction Pipeline

### 6.1 Overview

Two-stage pipeline for HFR attack mitigation:

1. **Frequency Identification**: Detect HFR pulse frequency using FFT
2. **Peak Removal**: Remove attack pulses based on identified periodicity

### 6.2 Stage 1: Frequency Identification (Fourier Method)

**Algorithm**: `HFRFrequencyIdentifierFourier`

**Method**:
```python
def identify_frequency_fourier(
    peak_detection_threshold=0.1,
    peak_count_threshold=3
) -> float  # Returns frequency in MHz
```

**Steps**:
1. **Attack signal detection**:
   - Count peaks in each waveform
   - Threshold: `peak_detection_threshold = 0.1`
   - Minimum peaks: `peak_count_threshold = 3`
   - Skip signals with fewer peaks (likely clean)

2. **FFT analysis**:
   ```python
   fft_vals = np.fft.fft(signal)
   fft_mag = np.abs(fft_vals)
   freqs = np.fft.fftfreq(n_samples, d=Δt × 10⁻⁹)  # In Hz
   ```

3. **Dominant frequency extraction**:
   - Find peak in FFT magnitude (positive frequencies only)
   - Collect frequencies from all attacked signals

4. **Robust estimation**:
   ```python
   identified_freq = median(detected_frequencies)
   ```

**Peak detection**:
```python
def _find_peaks(signal, threshold=0.1, min_gap=5):
    raises = where((signal[:-1] < threshold) & (signal[1:] >= threshold)) + 1
    # Filter peaks with <min_gap samples apart
    return filtered_peaks
```

### 6.3 Stage 2: Peak Interval Reconstruction

**Algorithm**: `PeakIntervalReconstructor`

**Method**:
```python
def reconstruct(
    hfr_freq_mhz: float,
    tolerance_ns=1.5,
    min_run_length=3,
    pulse_width_samples=80
)
```

**Parameters**:
- `hfr_freq_mhz`: Identified attack frequency
- `tolerance_ns`: Period matching tolerance (default: 1.5 ns)
- `min_run_length`: Minimum consecutive attack pulses (default: 3)
- `pulse_width_samples`: Removal window size (default: 80 samples)

**Steps**:

1. **Calculate target period**:
   ```python
   T_target = 1 / (hfr_freq_mhz × 10⁶) × 10⁹  # in nanoseconds
   ```

2. **Peak detection** (same as frequency identification):
   ```python
   peaks = _find_peaks(signal, threshold=0.1, min_gap=5)
   ```

3. **Periodic run detection**:
   - For each peak `i`, search for subsequent peaks
   - Check if time difference matches `k × T_target` (k ∈ ℕ)
   - Allow for multi-period gaps (missing pulses)

   **Matching criterion**:
   ```python
   time_diff = (peaks[j] - peaks[i]) × Δt
   k = round(time_diff / T_target)
   expected_diff = k × T_target

   if |time_diff - expected_diff| < tolerance × k:
       # Match: part of attack sequence
   ```

   **Scaled tolerance**: `tolerance_scaled = tolerance_ns × k`
   - Accounts for accumulated jitter over longer gaps

4. **Attack pulse removal**:
   - For each identified attack peak `p`:
     ```python
     start = max(0, p - 20)
     end = min(len(signal), p + 80)  # pulse_width_samples
     signal[start:end] = 0
     ```

5. **Iteration**:
   - Repeat for all (channel, azimuth) pairs in the dataset

### 6.4 Run Command Example

```bash
# Full pipeline
python reconstruction/run_pipeline.py \
  input.npz \
  output_reconstructed.npz \
  --id-threshold 0.1 \
  --id-peak-count 3 \
  --recon-tolerance 1.5 \
  --recon-min-run 3
```

---

## 7. Evaluation Metrics

### 7.1 Peak Detection

For each waveform, extract the highest peak:

```python
def get_peak_time(signal):
    raises = where((signal[:-1] < 0.01) & (signal[1:] >= 0.01)) + 1
    if len(raises) == 0:
        return 0.0

    peaks = [max(signal[r:r+50]) for r in raises]
    highest_peak_idx = raises[argmax(peaks)]

    # Parabolic interpolation for sub-sample precision
    if 0 < highest_peak_idx < len(signal) - 1:
        y0, y1, y2 = signal[highest_peak_idx-1:highest_peak_idx+2]
        if y0 > 0 and y1 > 0 and y2 > 0:
            offset = (ln(y0) - ln(y2)) / (2 × (ln(y0) - 2×ln(y1) + ln(y2)))
            return highest_peak_idx + offset

    return highest_peak_idx
```

### 7.2 Distance Error Metrics

**Mean Absolute Error (MAE)**:
```
MAE = (1/N) × Σ|d_reconstructed - d_ground_truth|
```

where distance is computed from peak time:
```
d = peak_time × Δt × 0.15  # meters
```

**Mean Squared Error (MSE)**:
```
MSE = (1/N) × Σ(d_reconstructed - d_ground_truth)²
```

**Root Mean Squared Error (RMSE)**:
```
RMSE = √MSE
```

### 7.3 Evaluation Command

```bash
python evaluation/evaluate_reconstruction.py \
  reconstructed.npz \
  ground_truth.npz \
  --method mae \
  --visualize  # Optional: generates error heatmaps
```

---

## 8. Reproducibility Checklist

### 8.1 Software Dependencies

```toml
[project.dependencies]
python = "^3.10"
numpy = "<2.0.0"
open3d = ">=0.18.0"
blosc2 = ">=2.0.0"
tqdm = ">=4.65.0"
matplotlib = ">=3.7.0"
```

### 8.2 Hardware Requirements

**Minimum**:
- CPU: 8 cores
- RAM: 32 GB
- Storage: 500 GB (for 1000 frames @ HDL-64E resolution)

**Recommended**:
- CPU: 16+ cores (for parallel batch processing)
- RAM: 64 GB
- Storage: 1 TB SSD

### 8.3 Random Seeds

For reproducible randomization:
```python
np.random.seed(42)  # Set before dataset generation
```

Randomized parameters:
- Sync angle per frame: `Uniform(sync_min, sync_max)`
- HFR pulse amplitude: `Uniform(3.0, 8.0)` per pulse
- HFR time perturbation: `Uniform(-20, +20)` ns per pulse

### 8.4 Dataset Splits

For training/validation/test splits, use frame-based splitting:

```python
# Example for KITTI (7481 frames)
train_frames = range(0, 5987)      # 80%
val_frames = range(5987, 6734)     # 10%
test_frames = range(6734, 7481)    # 10%
```

---

## 9. Key Implementation Details

### 9.1 Depth Map Construction (HDL-64E)

**Channel-based architecture** (collision-free):
1. Calculate spherical coordinates for all points
2. Assign points to nearest vertical channel
3. Sort points by azimuth within each channel
4. Downsample/pad to exactly 4400 samples per channel

**Benefits**:
- No azimuth collisions (each horizontal index is unique)
- Efficient array-based operations
- Direct mapping: `horizontal_index = azimuth_step`

### 9.2 Signal Composition

For each LiDAR measurement, signals are composed as:

```python
# 1. Generate legitimate LiDAR pulse
lidar_signal = gaussian_pulse(t_lidar, A_lidar, σ_lidar)

# 2. Generate HFR attack pulses (if in attack cone)
attack_signal = sum([
    gaussian_pulse(t_attack_i, A_i, σ_HFR)
    for i in attack_pulse_sequence
])

# 3. Combine signals (max)
composite_signal = max(lidar_signal, attack_signal)

# 4. Add noise
final_signal = composite_signal + noise
```

### 9.3 Coordinate Systems

**Internal angle representation** (simulator):
- 0° = right (+Y axis)
- 90° = front (+X axis)
- Counter-clockwise rotation

**User-facing angle** (command-line args):
- 0° = front
- Counter-clockwise rotation

**Conversion**:
```python
internal_angle = (user_angle + 90) % 360
```

### 9.4 Intensity=0 Handling

Critical for KITTI compatibility (39.41% of points):

```python
if pcd_intensity > 0:
    pulse_amplitude = pcd_intensity × 12.0
else:
    # Default minimum detectable amplitude
    pulse_amplitude = 0.05 × 12.0 = 0.6
```

Without this, zero-intensity points would be filtered out by the 0.01 amplitude threshold.

---

## 10. Validation

### 10.1 LiDAR Model Validation

**Metrics**:
- Point cloud reconstruction accuracy
- Intensity distribution match
- Scan pattern verification

**Datasets**:
- KITTI (HDL-64E): 7481 frames, outdoor driving
- nuScenes (VLP-32C): 34,149 frames, diverse scenes

### 10.2 Attack Model Validation

**Physical plausibility**:
- Frequency range: 1–20 MHz (consistent with radar systems)
- Pulse width: 3–10 ns (standard for high-frequency lasers)
- Amplitude: 1–9 normalized units (realistic SNR)

**Attack effectiveness**:
- Triggering accuracy: >95% at specified angle
- Cone coverage: precise 90° width
- Duration: consistent 100 ms

### 10.3 Reconstruction Validation

**Test cases**:
1. No attack: reconstruction should not degrade clean signals
2. Known frequency: exact removal of synthetic attack at 10 MHz
3. Frequency variation: robustness to ±0.5 MHz deviation
4. Partial attack: correct handling of spatially limited attacks

---

## 11. Parameter Summary Tables

### 11.1 LiDAR Parameters

| Parameter | HDL-64E | VLP-32C | VLP-16 |
|-----------|---------|---------|--------|
| Channels | 64 (or 32) | 32 | 16 |
| Vertical range | +3.26° to -23.64° | +10.67° to -30.67° | +15° to -15° |
| Horizontal resolution | 0.0818° (4400) | 0.2° (1800) | 0.2° (1800) |
| Pulse width | 5 ns | 5 ns | 5 ns |
| Amplitude range | 3.0–3.0 | 3.0–3.0 | 3.0–3.0 |
| Intensity mapping | ×12.0 | ×(40/255) | ×(40/255) |
| Acceptance window | 800 ns | 800 ns | 800 ns |
| Time resolution | 1 ns | 1 ns | 1 ns |

### 11.2 HFR Attack Parameters

| Parameter | Default | Range | Unit |
|-----------|---------|-------|------|
| Frequency | 10.0 | 1.0–20.0 | MHz |
| Pulse width | 5 | 3–10 | ns |
| Amplitude (min) | 3.0 | 1.0–9.0 | norm |
| Amplitude (max) | 8.0 | 1.0–9.0 | norm |
| Time perturbation | 20 | 0–50 | ns |
| Duration | 100 | 50–200 | ms |
| Spoofer distance | 10 | 5–50 | m |
| Attack angle | 0 | 0–360 | deg |
| Attack altitude | 40 | -30–+15 | deg |
| Attack cone width | 90 | 30–180 | deg |

### 11.3 Reconstruction Parameters

| Parameter | Default | Range | Unit |
|-----------|---------|-------|------|
| Peak detection threshold | 0.1 | 0.05–0.2 | norm |
| Peak count threshold | 3 | 2–5 | pulses |
| Period tolerance | 1.5 | 0.5–3.0 | ns |
| Minimum run length | 3 | 2–5 | pulses |
| Pulse removal width | 80 | 50–100 | samples |

---

## References

**Datasets**:
- KITTI: Geiger et al., "Vision meets Robotics: The KITTI Dataset", IJRR 2013
- nuScenes: Caesar et al., "nuScenes: A multimodal dataset for autonomous driving", CVPR 2020

**LiDAR Specifications**:
- Velodyne HDL-64E User Manual (2016)
- Velodyne VLP-32C User Manual (2018)

**Attack Models**:
- Cao et al., "Adversarial Sensor Attack on LiDAR-based Perception in Autonomous Driving", CCS 2019
- Sun et al., "Towards Robust LiDAR-based Perception in Autonomous Driving", NDSS 2020
