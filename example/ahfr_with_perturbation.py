from spaal2.core import (
    DummyOutdoor, 
    PreciseDuration, 
    apply_noise, 
    DummyLidarVLP16, 
    DummySpooferAdaptiveHFRWithPerturbation,
    visualize,  
)
import time

import numpy as np

def main():

    lidar = DummyLidarVLP16(amplitude=1.0, pulse_width=PreciseDuration(nanoseconds=5))
    outdoor = DummyOutdoor(50.0, 0.8)
    spoofer = DummySpooferAdaptiveHFRWithPerturbation(
        frequency=20 * 1e6, 
        duration=PreciseDuration(milliseconds=20),
        spoofer_distance_m=10.0,
        pulse_width=PreciseDuration(nanoseconds=5),
        perturbation_ns=20.0,  # Add perturbation
        amplitude=5.0,
    )

    start = time.time()
    point_list = []
    is_triggered = False
    while True:
        try:
            config, signal = lidar.scan()

            if config.altitude == 900 and abs(config.azimuth - 1000) < 20:
                spoofer.trigger(config, signal) # spooferをトリガー
                is_triggered = True


            signal = apply_noise(outdoor.apply(signal), ratio=0.1)
            legitimate_signal = signal.copy()
            external_signal = apply_noise(spoofer.get_range_signal(config.start_timestamp, config.accept_duration), ratio=0.01)
            
            signal = np.choose(
                signal < external_signal,
                [signal, external_signal]
            )
            signal = np.clip(signal, 0, 9)
            if is_triggered:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 4))
                plt.plot(signal, label="Signal")
                plt.title("Signal Visualization")
                plt.xlabel("Index")
                plt.ylabel("Signal Value")
                plt.legend()
                plt.grid()
                plt.show()
            points = lidar.receive(config, signal)
            point_list.extend(points)
            is_triggered = False

            
        except StopIteration:
            break

    print(f"There are {len(point_list)}/{lidar.max_index} points in total. ({1 - len(point_list) / lidar.max_index:.2%} points are lost)")
    print(f"Time elapsed: {(time.time() - start)*1000:.2f}ms")

    visualize(point_list)


if __name__ == "__main__":
    main()
