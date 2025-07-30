
import argparse
from hfr_frequency_identifier_fourier import HFRFrequencyIdentifierFourier
from peak_interval_reconstructor import PeakIntervalReconstructor

def run_reconstruction_pipeline(input_path: str, output_path: str, id_threshold: float, id_peak_count: int, recon_tolerance: float, recon_min_run: int):
    """
    Executes the full pipeline: identify frequency and then reconstruct the signal.
    """
    # --- Step 1: Identify HFR Frequency ---
    print(f"[Pipeline] Loading data from {input_path} to identify frequency...")
    identifier = HFRFrequencyIdentifierFourier(input_path)
    
    print("[Pipeline] Identifying HFR frequency using Fourier Transform...")
    hfr_frequency = identifier.identify_frequency_fourier(
        peak_detection_threshold=id_threshold,
        peak_count_threshold=id_peak_count
    )

    if hfr_frequency <= 0:
        print("[Pipeline] Could not identify a dominant HFR frequency. No reconstruction will be applied.")
        print("[Pipeline] Aborting.")
        return

    print(f"[Pipeline] Identified HFR frequency: {hfr_frequency:.2f} MHz")

    # --- Step 2: Apply Peak Interval Reconstruction ---
    print(f"[Pipeline] Loading data from {input_path} to reconstruct signal...")
    reconstructor = PeakIntervalReconstructor(input_path)
    
    print(f"[Pipeline] Reconstructing signal by removing peaks at {hfr_frequency:.2f} MHz period...")
    reconstructor.reconstruct(
        hfr_freq_mhz=hfr_frequency,
        tolerance_ns=recon_tolerance,
        min_run_length=recon_min_run
    )
    
    # --- Step 3: Save the result ---
    print(f"[Pipeline] Saving reconstructed data to {output_path}...")
    reconstructor.save(output_path)
    
    print("[Pipeline] Finished successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Full HFR attack reconstruction pipeline: Identify and Reconstruct.")
    parser.add_argument("input_npz", type=str, help="Path to the input .npz file with potential HFR attack.")
    parser.add_argument("output_npz", type=str, help="Path to save the cleaned .npz file.")
    
    # Arguments for frequency identification step
    parser.add_argument("--id-threshold", type=float, default=0.1, help="Peak detection threshold for identifying candidate signals.")
    parser.add_argument("--id-peak-count", type=int, default=3, help="Minimum number of peaks to consider a signal for FFT analysis.")

    # Arguments for peak interval reconstruction step
    parser.add_argument("--recon-tolerance", type=float, default=1.5, help="Tolerance for period matching in nanoseconds.")
    parser.add_argument("--recon-min-run", type=int, default=3, help="Minimum number of consecutive peaks to be considered an attack.")

    args = parser.parse_args()

    run_reconstruction_pipeline(
        input_path=args.input_npz,
        output_path=args.output_npz,
        id_threshold=args.id_threshold,
        id_peak_count=args.id_peak_count,
        recon_tolerance=args.recon_tolerance,
        recon_min_run=args.recon_min_run
    )
