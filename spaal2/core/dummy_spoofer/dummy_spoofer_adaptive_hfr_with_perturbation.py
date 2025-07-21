import numpy as np
import numpy.typing as npt
import math
from typing import Optional

from spaal2.core.dummy_spoofer import DummySpooferInterface
from spaal2.core.measurement_config import MeasurementConfig, PreciseDuration

class DummySpooferAdaptiveHFRWithPerturbation(DummySpooferInterface):
    """
    Adaptive HFR Spoofer with random time perturbation on each pulse.
    """
    def __init__(self,
                 frequency: float,
                 duration: PreciseDuration,
                 spoofer_distance_m: float,
                 pulse_width: PreciseDuration,
                 perturbation_ns: float,
                 amplitude: float = 9.0,
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
        self.amplitude = amplitude
        self.time_resolution_ns = time_resolution_ns
        self.pulse_period_ns = 1 / self.frequency * 1e9
        self.trigger_time: Optional[PreciseDuration] = None
        self.pulse_perturbations: dict[int, float] = {}

        self._precompute_pulse_shape()

    def set_amplitude(self, amplitude: float):
        self.amplitude = amplitude
        self._precompute_pulse_shape()

    def _precompute_pulse_shape(self):
        # Pre-calculate the Gaussian pulse shape at a high resolution (1.0 ns)
        sigma = self.pulse_width.in_nanoseconds / (2 * np.sqrt(2 * np.log2(2)))
        pulse_x_range = math.ceil(3 * sigma)
        pulse_x = np.arange(-pulse_x_range, pulse_x_range + 1, 1.0)
        self.pulse_shape = self.amplitude * np.exp(-(pulse_x ** 2) / (2 * sigma ** 2))

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
        if raises.size == 0:
            return
        peak_index = raises[0]
        peak_time_ns = peak_index * self.time_resolution_ns
        self.trigger_time = config.start_timestamp + PreciseDuration(nanoseconds=peak_time_ns)
        self.pulse_perturbations = {} # Reset perturbations on trigger
        print(f"Triggered at {self.trigger_time.in_nanoseconds}ns")

    def get_range_signal(self, start_timestamp: PreciseDuration, duration: PreciseDuration) -> npt.NDArray[np.float64]:
        output_length = int(duration.in_nanoseconds / self.time_resolution_ns)
        if self.trigger_time is None:
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
            return np.zeros(output_length)

        # Create the target time points for the output signal
        target_time_points = request_start_ns + np.arange(output_length) * self.time_resolution_ns
        output_signal = np.zeros(output_length)

        pulse_half_width_ns = (len(self.pulse_shape) // 2)

        start_pulse_idx = math.floor((request_start_ns - attack_start_ns - pulse_half_width_ns - self.perturbation_ns) / self.pulse_period_ns)
        end_pulse_idx = math.ceil((request_end_ns - attack_start_ns + pulse_half_width_ns + self.perturbation_ns) / self.pulse_period_ns)

        for pulse_idx in range(start_pulse_idx, end_pulse_idx):
            ideal_pulse_time_ns = attack_start_ns + pulse_idx * self.pulse_period_ns
            
            if pulse_idx not in self.pulse_perturbations:
                self.pulse_perturbations[pulse_idx] = np.random.uniform(-self.perturbation_ns, self.perturbation_ns)
            perturbation = self.pulse_perturbations[pulse_idx]
            perturbed_time_ns = ideal_pulse_time_ns + perturbation

            if not (attack_start_ns <= perturbed_time_ns < attack_end_ns):
                continue

            # Define the time window for this specific pulse
            pulse_start_ns = perturbed_time_ns - pulse_half_width_ns
            pulse_end_ns = perturbed_time_ns + pulse_half_width_ns

            # Find which points in the target signal are affected by this pulse
            affected_indices = np.where((target_time_points >= pulse_start_ns) & (target_time_points <= pulse_end_ns))[0]

            if len(affected_indices) > 0:
                # Time points relative to the center of the pulse
                relative_time_points = target_time_points[affected_indices] - perturbed_time_ns
                # Interpolate the high-res pulse_shape to find the values at our target time points
                high_res_x = np.arange(-pulse_half_width_ns, pulse_half_width_ns + 1, 1.0)
                interpolated_values = np.interp(relative_time_points, high_res_x, self.pulse_shape)
                output_signal[affected_indices] += interpolated_values

        return np.clip(output_signal, 0, self.amplitude)