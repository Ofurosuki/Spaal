
from torch.utils.data import DataLoader
from dataset import HistMatrixDataset

if __name__ == '__main__':
    # 1. Create an instance of the dataset
    # Replace this with the actual path to your generated dataset
    dataset_root = './my_new_dataset' 

    print(f"Loading dataset from: {dataset_root}")
    try:
        hist_dataset = HistMatrixDataset(dataset_root_path=dataset_root)
        print(f"Dataset loaded successfully. Found {len(hist_dataset)} samples.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset_root path is correct and the dataset has been generated.")
        exit()

    # 2. Create a DataLoader
    # You can adjust batch_size, shuffle, num_workers, etc. as needed
    data_loader = DataLoader(hist_dataset, batch_size=3, shuffle=True, num_workers=0)

    # 3. Iterate over the data
    print("\nIterating through the DataLoader...")
    for i, batch in enumerate(data_loader):
        # Each 'batch' is a dictionary containing a batch of signals and labels
        signals_batch = batch['signal']
        labels_batch = batch['labels']

        print(f"Batch {i+1}:")
        print(f"  Signals batch shape: {signals_batch.shape}")
        print(f"  Labels batch shape: {labels_batch.shape}")
        
        # Here you would typically pass the data to your model
        # e.g., output = model(signals_batch)
        
        # We'll just show the first batch for this example
        if i == 0:
            break
    
    print("\nExample finished.")

