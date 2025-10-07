import numpy as np
import numpy.typing as npt
import math
from typing import Optional

from spaal2.core.dummy_spoofer import DummySpooferInterface
from spaal2.core.measurement_config import MeasurementConfig, PreciseDuration

class DummySpooferAdaptiveHFRWithPerturbation(DummySpooferInterface):
    """
    Adaptive HFR Spoofer with random time perturbation and random amplitude on each pulse.
    """
    def __init__(self,
                 frequency: float,
                 duration: PreciseDuration,
                 spoofer_distance_m: float,
                 pulse_width: PreciseDuration,
                 perturbation_ns: float,
                 time_resolution_ns: float = 1.0,
                 debug: bool = False) -> None:
        """
        Parameters
        ----------
        frequency : float
            The frequency of the pulses in Hz.
        duration : PreciseDuration
            The duration of the attack after being triggered.
        spoofer_distance_m : float
            The distance between the spoofer and the LiDAR in meters.
        pulse_width : PreciseDuration
            The width of a single pulse.
        perturbation_ns : float
            The maximum random time perturbation to add to each pulse, in nanoseconds.
            The perturbation will be in the range [-perturbation_ns, +perturbation_ns].
        time_resolution_ns : float, optional
            The time resolution of the output signal in nanoseconds, by default 1.0.
        debug : bool, optional
            Whether to print debug information, by default False.
        """
        self.frequency = frequency
        self.duration = duration
        self.distance_m = spoofer_distance_m
        self.pulse_width = pulse_width
        self.perturbation_ns = perturbation_ns
        self.time_resolution_ns = time_resolution_ns
        self.pulse_period_ns = 1 / self.frequency * 1e9
        self.trigger_time: Optional[PreciseDuration] = None
        self.pulse_perturbations: dict[int, float] = {}
        
        # Amplitude settings
        self.amplitude_range: tuple[float, float] = (9.0, 9.0) # Default range
        self.amplitude_sequence: Optional[np.ndarray] = None

        self._precompute_normalized_pulse_shape()

    def set_amplitude_range(self, amplitude_range: tuple[float, float]):
        """Sets the range for random pulse amplitudes."""
        self.amplitude_range = amplitude_range

    def _precompute_normalized_pulse_shape(self):
        # Pre-calculate the Gaussian pulse shape with amplitude 1.0
        sigma = self.pulse_width.in_nanoseconds / (2 * np.sqrt(2 * np.log2(2)))
        pulse_x_range = math.ceil(3 * sigma)
        pulse_x = np.arange(-pulse_x_range, pulse_x_range + 1, 1.0)
        # The pulse shape is normalized to have a peak amplitude of 1.0
        self.normalized_pulse_shape = np.exp(-(pulse_x ** 2) / (2 * sigma ** 2))

    def trigger(self, config: MeasurementConfig, signal: npt.NDArray[np.float64]):
        if self.trigger_time is not None:
            return

        delay_indices = int(self.distance_m / (0.15 * self.time_resolution_ns))
        new_signal = np.zeros_like(signal)
        if delay_indices < len(signal):
            new_signal[delay_indices:] = signal[:-delay_indices]
        signal = new_signal

        raises = np.flatnonzero(
            (signal[:-1] < 0.5) & (signal[1:] >= 0.5)
        ) + 1
        
        peak_time_ns: float
        if raises.size > 0:
            # 基準パルスが見つかった場合：その時刻を同期に使う
            peak_index = raises[0]
            peak_time_ns = peak_index * self.time_resolution_ns
        else:
            # 基準パルスが見つからない場合：spoofer自身の距離からToFを計算して代替する
            #print("Spoofer trigger: No valid pulse found. Using spoofer's own distance for fallback timing.")
            # 光速 (m/ns)
            SPEED_OF_LIGHT_M_PER_NS = 0.299792458 
            # ToF(往復時間) = 2 * 距離 / 光速
            time_of_flight_ns = (self.distance_m * 2) / SPEED_OF_LIGHT_M_PER_NS
            peak_time_ns = time_of_flight_ns

        self.trigger_time = config.start_timestamp + PreciseDuration(nanoseconds=peak_time_ns)
        
        # Reset state for the new attack
        self.pulse_perturbations = {}
        
        # Generate the random amplitude sequence for the entire attack duration
        num_pulses = math.ceil(self.duration.in_nanoseconds / self.pulse_period_ns)
        self.amplitude_sequence = np.random.uniform(
            low=self.amplitude_range[0],
            high=self.amplitude_range[1],
            size=num_pulses
        )
        #print(f"Triggered at {self.trigger_time.in_nanoseconds}ns. Generated {num_pulses} random amplitudes.")

    def get_range_signal(self, start_timestamp: PreciseDuration, duration: PreciseDuration) -> npt.NDArray[np.float64]:
        output_length = int(duration.in_nanoseconds / self.time_resolution_ns)
        if self.trigger_time is None or self.amplitude_sequence is None:
            return np.zeros(output_length)

        attack_start_time = self.trigger_time
        attack_end_time = self.trigger_time + self.duration

        request_start_ns = start_timestamp.in_nanoseconds
        request_end_ns = request_start_ns + duration.in_nanoseconds

        attack_start_ns = attack_start_time.in_nanoseconds
        attack_end_ns = attack_end_time.in_nanoseconds

        if request_end_ns <= attack_start_ns or request_start_ns >= attack_end_ns:
            if request_start_ns >= attack_end_ns:
                self.trigger_time = None
                self.amplitude_sequence = None # Clear sequence after attack
            return np.zeros(output_length)

        target_time_points = request_start_ns + np.arange(output_length) * self.time_resolution_ns
        output_signal = np.zeros(output_length)

        pulse_half_width_ns = (len(self.normalized_pulse_shape) // 2)

        start_pulse_idx = math.floor((request_start_ns - attack_start_ns - pulse_half_width_ns - self.perturbation_ns) / self.pulse_period_ns)
        end_pulse_idx = math.ceil((request_end_ns - attack_start_ns + pulse_half_width_ns + self.perturbation_ns) / self.pulse_period_ns)

        for pulse_idx in range(start_pulse_idx, end_pulse_idx):
            if not (0 <= pulse_idx < len(self.amplitude_sequence)):
                continue # Skip pulses outside the generated sequence range

            current_amplitude = self.amplitude_sequence[pulse_idx]
            ideal_pulse_time_ns = attack_start_ns + pulse_idx * self.pulse_period_ns
            
            if pulse_idx not in self.pulse_perturbations:
                self.pulse_perturbations[pulse_idx] = np.random.uniform(-self.perturbation_ns, self.perturbation_ns)
            perturbation = self.pulse_perturbations[pulse_idx]
            perturbed_time_ns = ideal_pulse_time_ns + perturbation

            if not (attack_start_ns <= perturbed_time_ns < attack_end_ns):
                continue

            pulse_start_ns = perturbed_time_ns - pulse_half_width_ns
            pulse_end_ns = perturbed_time_ns + pulse_half_width_ns

            affected_indices = np.where((target_time_points >= pulse_start_ns) & (target_time_points <= pulse_end_ns))[0]

            if len(affected_indices) > 0:
                relative_time_points = target_time_points[affected_indices] - perturbed_time_ns
                high_res_x = np.arange(-pulse_half_width_ns, pulse_half_width_ns + 1, 1.0)
                
                # Interpolate using the normalized pulse shape and scale by the current pulse's unique amplitude
                interpolated_values = np.interp(relative_time_points, high_res_x, self.normalized_pulse_shape) * current_amplitude
                output_signal[affected_indices] += interpolated_values

        return np.clip(output_signal, 0, max(self.amplitude_range))
