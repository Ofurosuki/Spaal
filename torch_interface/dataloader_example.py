import os
import time
from torch.utils.data import DataLoader
from dataset import HistMatrixDataset

if __name__ == '__main__':
    # 1. Define dataset parameters
    # IMPORTANT: Replace this with the absolute root path of your dataset structure.
    #root = 'C:/Users/nextr/spaal2-core/my_new_dataset' 
    root = '/data2/yoshida/1010_dataset'  # Example for Linux/Mac
    split = 'train'
    dataset_name = 'nuscenes'
    scan_type = 'horizontal'
    sync_angle = 1

    # 2. Create an instance of the dataset
    sync_angle_str = str(sync_angle).replace('.', '_')
    dataset_path = os.path.join(root, split, dataset_name, scan_type, sync_angle_str)
    print(f"Loading dataset from: {dataset_path}")
    
    try:
        hist_dataset = HistMatrixDataset(
            root_path=root,
            split=split,
            dataset_name=dataset_name,
            scan_type=scan_type,
            sync_angle=sync_angle
        )
        print(f"Dataset loaded successfully. Found {len(hist_dataset)} samples.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset path is correct and the dataset has been generated.")
        exit()

    # 3. Create a DataLoader
    data_loader = DataLoader(hist_dataset, batch_size=4, shuffle=True, num_workers=0)

    # 4. Iterate over the data and measure time
    print("\nIterating through the DataLoader to measure batch loading time...")
    
    start_time = time.time()
    
    for i, batch in enumerate(data_loader):
        # The time measurement is taken as soon as the first batch is yielded
        end_time = time.time()
        
        signals_batch = batch['signal']
        labels_batch = batch['labels']

        print(f"Time to load first batch: {end_time - start_time:.4f} seconds")
        
        print(f"Batch {i+1}:")
        print(f"  Signals batch shape: {signals_batch.shape}")
        print(f"  Labels batch shape: {labels_batch.shape}")
        
        # We'll just show the first batch for this example
        break
    
    print("\nExample finished.")