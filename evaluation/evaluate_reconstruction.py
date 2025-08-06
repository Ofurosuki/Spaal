import numpy as np
import argparse
import matplotlib.pyplot as plt

class ReconstructionEvaluator:
    def __init__(self, hist_matrix: np.ndarray, answer_matrix: np.ndarray, evaluation_method: str = "mae", visualize: bool = False):
        self.hist_matrix = hist_matrix
        self.answer_matrix = answer_matrix
        self.evaluation_method = self._get_evaluation_method(evaluation_method)
        self.visualize = visualize

    def _get_peak_index_from_signal(self, signal: np.ndarray) -> int:
        """
        Finds the index of the highest peak in a given signal by detecting rises.
        """
        if np.all(signal < 0.01):
            return 0

        raises = np.flatnonzero(
            (signal[:-1] < 0.01) & (signal[1:] >= 0.01)
        ) + 1
        
        if len(raises) == 0:
            # If no rise is detected, return the absolute max index as a fallback.
            return np.argmax(signal)

        peaks = np.empty_like(raises, dtype=np.float64)
        for i in range(len(raises)):
            peaks[i] = np.max(
                signal[raises[i]:min(len(signal), raises[i] + 50)]
            )

        highest_peak_index_in_peaks = np.argmax(peaks)
        highest_peak_time = raises[highest_peak_index_in_peaks]

        return highest_peak_time

    def _mean_absolute_error(self, reconstructed_distances: np.ndarray, true_distances: np.ndarray) -> float:
        """
        Calculates the mean absolute error between reconstructed and true distances.
        """
        return np.mean(np.abs(reconstructed_distances - true_distances))

    def _mean_squared_error(self, reconstructed_distances: np.ndarray, true_distances: np.ndarray) -> float:
        """
        Calculates the mean squared error between reconstructed and true distances.
        """
        return np.mean((reconstructed_distances - true_distances) ** 2)

    def _get_evaluation_method(self, method_name: str):
        if method_name == "mae":
            return self._mean_absolute_error
        elif method_name == "mse":
            return self._mean_squared_error
        else:
            raise ValueError(f"Unknown evaluation method: {method_name}")

    def _visualize_error_heatmap(self, error_matrix: np.ndarray, frame_num: int):
        """
        Visualizes the error matrix as a heatmap and saves it to a file.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(error_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Absolute Error')
        plt.xlabel("Horizontal Step")
        plt.ylabel("Channel")
        plt.title(f"Reconstruction Error Heatmap (Frame {frame_num})")
        plt.savefig(f"reconstruction_error_heatmap_frame_{frame_num}.png")
        plt.close()

    def evaluate(self) -> float:
        """
        Evaluates the reconstruction accuracy.
        """
        num_frames, num_channels, num_horizontal_steps, _ = self.hist_matrix.shape
        
        reconstructed_distances = np.zeros((num_frames, num_channels, num_horizontal_steps))
        
        for frame in range(num_frames):
            for channel in range(num_channels):
                for step in range(num_horizontal_steps):
                    signal = self.hist_matrix[frame, channel, step]
                    reconstructed_distances[frame, channel, step] = self._get_peak_index_from_signal(signal)
            
            if self.visualize:
                error_matrix = np.abs(reconstructed_distances[frame] - self.answer_matrix[frame])
                self._visualize_error_heatmap(error_matrix, frame)
                    
        return self.evaluation_method(reconstructed_distances, self.answer_matrix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate reconstruction accuracy.")
    parser.add_argument("reconstructed_hist_matrix", help="Path to the reconstructed hist-matrix .npz file.")
    parser.add_argument("answer_matrix", help="Path to the answer-matrix .npz file.")
    parser.add_argument("--method", type=str, default="mae", choices=["mae", "mse"], help="Evaluation method to use.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the error as a heatmap.")
    
    args = parser.parse_args()
    
    reconstructed_data = np.load(args.reconstructed_hist_matrix)
    answer_data = np.load(args.answer_matrix)
    
    hist_matrix = reconstructed_data['signals']
    answer_matrix = answer_data['answer_matrix']
    
    evaluator = ReconstructionEvaluator(hist_matrix, answer_matrix, evaluation_method=args.method, visualize=args.visualize)
    error = evaluator.evaluate()
    
    print(f"Reconstruction error ({args.method}): {error}")
