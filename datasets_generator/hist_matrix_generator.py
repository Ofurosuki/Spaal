import numpy as np
import os
import argparse
from spaal2.core import (
    PreciseDuration, DummyOutdoor, apply_noise, gen_sunlight,
)
from spaal2.core.dummy_lidar.dummy_lidar_vlp16 import DummyLidarVLP16
from spaal2.core.dummy_lidar.dummy_lidar_vlp16_pcd import PcdLidarVLP16
from spaal2.core.dummy_spoofer.dummy_spoofer_adaptive_hfr_with_perturbation import DummySpooferAdaptiveHFRWithPerturbation
from spaal2.core.dummy_spoofer.dummy_spoofer_off import DummySpooferOff

class LidarSignalDatasetGenerator:
    def __init__(self, 
                 lidar_type: str = "VLP16",
                 pcd_file: str = None,
                 output_dir: str = "./datasets",
                 outdoor_distance: float = 50.0, outdoor_ratio: float = 0.8,
                 spoofer_type: str = "adaptive_hfr_perturbation",
                 spoofer_frequency: float = 10 * 1e6,
                 spoofer_duration_ms: float = 20,
                 spoofer_distance_m: float = 10.0,
                 spoofer_pulse_width_ns: float = 5,
                 spoofer_perturbation_ns: float = 0.0,
                 spoofer_amplitude_range: tuple[float, float] = (5.0, 9.0),
                 lidar_amplitude_range: tuple[float, float] = (1.0, 7.0),
                 lidar_pulse_width_ns: float = 5,
                 time_resolution_ns: float = 1.0,
                 noise_ratio: float = 0.1,
                 sunlight_mean: float = 0.5):

        self.lidar_type = lidar_type
        self.pcd_file = pcd_file
        self.time_resolution_ns = time_resolution_ns

        if self.lidar_type == "VLP16":
            self.lidar = DummyLidarVLP16(
                amplitude=lidar_amplitude_range[0], 
                pulse_width=PreciseDuration(nanoseconds=lidar_pulse_width_ns),
                time_resolution_ns=self.time_resolution_ns
            )
            self.channels = 16
            self.horizontal_resolution = 1800
        elif self.lidar_type == "PCD_VLP16":
            if not self.pcd_file or not os.path.exists(self.pcd_file):
                raise ValueError(f"PCD file path must be provided and valid for PCD_VLP16 lidar type. Provided: {self.pcd_file}")
            self.lidar = PcdLidarVLP16(
                pcd_file_path=self.pcd_file,
                lidar_position=np.array([0.0, 0.0, 0.0]),
                lidar_rotation=np.array([0.0, 0.0, 0.0]),
                amplitude=lidar_amplitude_range[0],
                pulse_width=PreciseDuration(nanoseconds=lidar_pulse_width_ns),
                time_resolution_ns=self.time_resolution_ns
            )
            self.channels = 16
            self.horizontal_resolution = self.lidar.max_index // self.channels
        else:
            raise ValueError(f"Unknown LiDAR model: {lidar_type}")

        self.samples_per_scan = int(self.lidar.accept_window.in_nanoseconds / self.lidar.time_resolution_ns)
        
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.outdoor = DummyOutdoor(outdoor_distance, outdoor_ratio)
        self.noise_ratio = noise_ratio
        self.sunlight_mean = sunlight_mean
        self.lidar_amplitude_range = lidar_amplitude_range
        self.spoofer_amplitude_range = spoofer_amplitude_range
        self.spoofer_type = spoofer_type

        if self.spoofer_type == "adaptive_hfr_perturbation":
            self.spoofer = DummySpooferAdaptiveHFRWithPerturbation(
                frequency=spoofer_frequency,
                duration=PreciseDuration(milliseconds=spoofer_duration_ms),
                spoofer_distance_m=spoofer_distance_m,
                pulse_width=PreciseDuration(nanoseconds=spoofer_pulse_width_ns),
                perturbation_ns=spoofer_perturbation_ns,
                amplitude=spoofer_amplitude_range[0],
                time_resolution_ns=self.lidar.time_resolution_ns
            )
        elif self.spoofer_type == "off":
            self.spoofer = DummySpooferOff()
        else:
            raise ValueError(f"Spoofer type {self.spoofer_type} not implemented for this script yet.")

    def generate(self, num_frames: int, filename_prefix: str = "lidar_signal"):
        all_frames_data = np.zeros((num_frames, self.channels, self.horizontal_resolution, self.samples_per_scan), dtype=np.float32)

        for frame_num in range(num_frames):
            print(f"Generating frame {frame_num + 1}/{num_frames}...")
            current_lidar = self.lidar.new_frame(base_timestamp=PreciseDuration(nanoseconds=frame_num * 10**9))
            frame_data = np.zeros((self.channels, self.horizontal_resolution, self.samples_per_scan), dtype=np.float32)

            try:
                for i in range(current_lidar.max_index):
                    config, signal = current_lidar.scan()

                    lidar_amp = np.random.uniform(self.lidar_amplitude_range[0], self.lidar_amplitude_range[1])
                    current_lidar.set_amplitude(lidar_amp)
                    
                    if self.spoofer_type != "off":
                        spoofer_amp = np.random.uniform(self.spoofer_amplitude_range[0], self.spoofer_amplitude_range[1])
                        self.spoofer.set_amplitude(spoofer_amp)

                        if config.altitude == 900 and config.azimuth == 7000:
                            self.spoofer.trigger(config, signal)
                        
                        external_signal = apply_noise(self.spoofer.get_range_signal(config.start_timestamp, config.accept_duration), ratio=0.01)
                        signal = np.maximum(signal, external_signal)

                    signal = np.clip(signal, 0, 9)

                    horizontal_index = (current_lidar.index -1) // self.channels
                    vertical_index = (current_lidar.index - 1) % self.channels

                    if horizontal_index < self.horizontal_resolution:
                        frame_data[vertical_index, horizontal_index, :] = signal

            except StopIteration:
                pass

            all_frames_data[frame_num, :, :, :] = frame_data

        initial_azimuth_offset = self.lidar.initial_azimuth_offset if hasattr(self.lidar, 'initial_azimuth_offset') else 0.0
        vertical_angles = self.lidar.vertical_angles
        fov = 360.0  # FOV for VLP16 is 360 degrees

        output_filename = os.path.join(self.output_dir, f"{filename_prefix}.npz")
        np.savez(output_filename, 
                 signals=all_frames_data, 
                 initial_azimuth_offset=initial_azimuth_offset, 
                 vertical_angles=vertical_angles,
                 fov=fov,
                 time_resolution_ns=self.time_resolution_ns)
        print(f"Saved all frames to {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate LiDAR signal datasets.")
    parser.add_argument("--lidar-type", type=str, default="VLP16", choices=["VLP16", "PCD_VLP16"],
                        help="Type of LiDAR to use.")
    parser.add_argument("--pcd-file", type=str, default=None,
                        help="Path to the PCD file, required if lidar-type is PCD_VLP16.")
    parser.add_argument("--num-frames", type=int, default=1,
                        help="Number of frames to generate.")
    parser.add_argument("--output-dir", type=str, default="./lidar_datasets",
                        help="Directory to save the generated dataset.")
    parser.add_argument("--time-resolution-ns", type=float, default=1.0,
                        help="Time resolution in nanoseconds for the simulation.")
    parser.add_argument("--spoofer-type", type=str, default="adaptive_hfr_perturbation", choices=["adaptive_hfr_perturbation", "off"],
                        help="Type of spoofer to use.")

    args = parser.parse_args()

    if args.lidar_type == "PCD_VLP16" and not args.pcd_file:
        parser.error("--pcd-file is required when --lidar-type is PCD_VLP16")

    generator = LidarSignalDatasetGenerator(
        lidar_type=args.lidar_type,
        pcd_file=args.pcd_file,
        output_dir=args.output_dir,
        time_resolution_ns=args.time_resolution_ns,
        spoofer_type=args.spoofer_type
    )
    generator.generate(num_frames=args.num_frames)
