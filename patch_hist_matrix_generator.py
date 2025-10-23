#!/usr/bin/env python3
"""
Patch hist_matrix_generator.py to support different LiDAR types with appropriate defaults
"""

file_path = "datasets_generator/hist_matrix_generator.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Modification 1: Change __init__ signature to use None for output_horizontal_resolution_deg
content = content.replace(
    'output_horizontal_resolution_deg: float = 0.0818):',
    'output_horizontal_resolution_deg: float = None):'
)

# Modification 2: Change horizontal_resolution_deg default from 0.2 to 0.1
content = content.replace(
    'horizontal_resolution_deg: float = 0.2,',
    'horizontal_resolution_deg: float = 0.1,'
)

# Modification 3: Add auto-detection logic after __init__ parameters
# Find the line "self.lidar_type = lidar_type" and add logic after it
old_init_block = '''        self.lidar_type = lidar_type
        self.pcd_directory = pcd_directory
        self.json_path = json_path
        self.time_resolution_ns = time_resolution_ns'''

new_init_block = '''        self.lidar_type = lidar_type

        # Auto-detect output_horizontal_resolution_deg if not provided
        if output_horizontal_resolution_deg is None:
            if lidar_type == "PCD_HDL64E":
                output_horizontal_resolution_deg = 0.0818  # 4400 samples
            elif lidar_type in ["PCD_VLP32c", "PCD_VLP16", "VLP16"]:
                output_horizontal_resolution_deg = 0.2  # 1800 samples
            else:
                output_horizontal_resolution_deg = 0.2  # Default

        self.pcd_directory = pcd_directory
        self.json_path = json_path
        self.time_resolution_ns = time_resolution_ns'''

content = content.replace(old_init_block, new_init_block)

# Modification 4: Update argparse default to None
old_argparse = '''    parser.add_argument("--output-horizontal-resolution-deg", type=float, default=0.2,
                        help="Output horizontal resolution in degrees for HDL-64E (determines samples per channel). Default is 0.0818 degrees (4400 samples).")'''

new_argparse = '''    parser.add_argument("--output-horizontal-resolution-deg", type=float, default=None,
                        help="Output horizontal resolution in degrees (determines samples per channel). Auto-detected if not specified: HDL-64E=0.0818° (4400 samples), VLP32c/VLP16=0.2° (1800 samples).")'''

content = content.replace(old_argparse, new_argparse)

# Modification 5: Update horizontal-resolution-deg default from 0.2 to 0.1
old_h_res = '''    parser.add_argument("--horizontal-resolution-deg", type=float, default=0.1,
                        help="Internal horizontal resolution in degrees for HDL-64E. Default is 0.1 degrees.")'''

new_h_res = '''    parser.add_argument("--horizontal-resolution-deg", type=float, default=0.1,
                        help="Internal horizontal resolution in degrees for PCD-based LiDARs. Default is 0.1 degrees.")'''

content = content.replace(old_h_res, new_h_res)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully patched hist_matrix_generator.py")
print("\nChanges made:")
print("1. output_horizontal_resolution_deg default changed to None (auto-detect)")
print("2. Auto-detection logic added: HDL-64E=0.0818°, VLP32c/VLP16=0.2°")
print("3. horizontal_resolution_deg default remains 0.1°")
print("4. Command-line help text updated")
