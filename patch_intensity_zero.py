#!/usr/bin/env python3
"""
Patch dummy_lidar_hdl64e.py to handle intensity=0 points
"""

file_path = "spaal2/core/dummy_lidar/dummy_lidar_hdl64e.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and replace the two occurrences
modified = False
i = 0
while i < len(lines):
    # First occurrence (Gaussian pulse)
    if (i + 2 < len(lines) and
        "pulse_amplitude = self.amplitude" in lines[i] and
        "if pcd_intensity is not None:" in lines[i+1] and
        "pulse_amplitude = pcd_intensity * self.intensity_to_amplitude_ratio" in lines[i+2] and
        "# Generate the Gaussian pulse" in lines[i+4]):

        print(f"Found first occurrence at line {i+1}")
        # Replace lines[i+1:i+3] with modified code
        indent = " " * 28
        new_code = [
            f"{indent}if pcd_intensity is not None:\n",
            f"{indent}    if pcd_intensity > 0:\n",
            f"{indent}        pulse_amplitude = pcd_intensity * self.intensity_to_amplitude_ratio\n",
            f"{indent}    else:\n",
            f"{indent}        # intensity=0の場合、デフォルト値を使用（検出可能な最小値）\n",
            f"{indent}        pulse_amplitude = 0.05 * self.intensity_to_amplitude_ratio\n",
        ]
        lines[i+1:i+3] = new_code
        modified = True
        i += len(new_code) + 1
        continue

    # Second occurrence (fallback for zero pulse width)
    if (i + 2 < len(lines) and
        "pulse_amplitude = self.amplitude" in lines[i] and
        "if pcd_intensity is not None:" in lines[i+1] and
        "pulse_amplitude = pcd_intensity * self.intensity_to_amplitude_ratio" in lines[i+2] and
        "signal[time_of_flight_index] += pulse_amplitude" in lines[i+3]):

        print(f"Found second occurrence at line {i+1}")
        # Replace lines[i+1:i+3] with modified code
        indent = " " * 24
        new_code = [
            f"{indent}if pcd_intensity is not None:\n",
            f"{indent}    if pcd_intensity > 0:\n",
            f"{indent}        pulse_amplitude = pcd_intensity * self.intensity_to_amplitude_ratio\n",
            f"{indent}    else:\n",
            f"{indent}        # intensity=0の場合、デフォルト値を使用（検出可能な最小値）\n",
            f"{indent}        pulse_amplitude = 0.05 * self.intensity_to_amplitude_ratio\n",
        ]
        lines[i+1:i+3] = new_code
        modified = True
        i += len(new_code) + 1
        continue

    i += 1

if modified:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"\nSuccessfully patched {file_path}")
else:
    print("No matching code found to patch")
