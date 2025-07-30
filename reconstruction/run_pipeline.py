
import argparse
from hfr_frequency_identifier_fourier import HFRFrequencyIdentifierFourier
from notch_filter import NotchFilter

def run_reconstruction_pipeline(input_path: str, output_path: str, id_threshold: float, id_peak_count: int, filter_bw: float, filter_harmonics: int):
    """
    Executes the full pipeline: identify frequency and then apply a notch filter.
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
        print("[Pipeline] Could not identify a dominant HFR frequency. No filtering will be applied.")
        # In a real scenario, you might want to copy the input to output or handle this differently.
        print("[Pipeline] Aborting.")
        return

    print(f"[Pipeline] Identified HFR frequency: {hfr_frequency:.2f} MHz")

    # --- Step 2: Apply Notch Filter ---
    # The NotchFilter class loads the file again. This is slightly inefficient 
    # but keeps the classes nicely decoupled as self-contained units.
    print(f"[Pipeline] Loading data from {input_path} to apply filter...")
    notch_filter = NotchFilter(input_path)
    
    print(f"[Pipeline] Applying notch filter to remove {hfr_frequency:.2f} MHz...")
    notch_filter.apply_filter(
        hfr_freq_mhz=hfr_frequency,
        bandwidth_mhz=filter_bw,
        harmonics=filter_harmonics
    )
    
    # --- Step 3: Save the result ---
    print(f"[Pipeline] Saving filtered data to {output_path}...")
    notch_filter.save(output_path)
    
    print("[Pipeline] Finished successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Full HFR attack reconstruction pipeline: Identify and Filter.")
    parser.add_argument("input_npz", type=str, help="Path to the input .npz file with potential HFR attack.")
    parser.add_argument("output_npz", type=str, help="Path to save the cleaned .npz file.")
    
    # Arguments for frequency identification step
    parser.add_argument("--id-threshold", type=float, default=0.1, help="Peak detection threshold for identifying candidate signals.")
    parser.add_argument("--id-peak-count", type=int, default=3, help="Minimum number of peaks to consider a signal for FFT analysis.")

    # Arguments for notch filter step
    parser.add_argument("--filter-bw", type=float, default=0.5, help="Bandwidth of the notch filter in MHz.")
    parser.add_argument("--filter-harmonics", type=int, default=3, help="Number of harmonics to remove.")

    args = parser.parse_args()

    run_reconstruction_pipeline(
        input_path=args.input_npz,
        output_path=args.output_npz,
        id_threshold=args.id_threshold,
        id_peak_count=args.id_peak_count,
        filter_bw=args.filter_bw,
        filter_harmonics=args.filter_harmonics
    )
