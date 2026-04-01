import os
import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

class ECGDataLoader:
    """Handles MIT-BIH dataset downloading, processing, and loading."""
    
    def __init__(self, raw_dir, processed_dir, sampling_rate=360):
        self.raw_dir = os.path.abspath(raw_dir)
        self.processed_dir = os.path.abspath(processed_dir)
        self.sampling_rate = sampling_rate
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def download_mitdb(self):
        """Downloads the MIT-BIH Arrhythmia Database."""
        if os.path.exists(os.path.join(self.raw_dir, '100.dat')):
            print("MIT-BIH dataset already exists.")
            return
        print(f"Downloading mitdb to {self.raw_dir}...")
        wfdb.dl_database('mitdb', self.raw_dir)
        print("Download complete.")

    def process_data(self, force=False):
        """Processes raw records into numpy arrays and saves them."""
        x_path = os.path.join(self.processed_dir, 'X.npy')
        y_path = os.path.join(self.processed_dir, 'y.npy')
        
        if not force and os.path.exists(x_path):
            print("Processed data already exists. Use force=True to re-process.")
            return

        print("Processing records...")
        all_files = os.listdir(self.raw_dir)
        record_names = sorted(list(set([f.split('.')[0] for f in all_files if f.endswith('.dat')])))
        
        if not record_names:
            print("No records found. Cannot process data.")
            return

        X, y = self._process_records(record_names)
        
        # Save processed data
        np.save(x_path, X)
        np.save(y_path, y)
        print(f"Processed {len(X)} heartbeats and saved to {self.processed_dir}")

    def get_loaders(self, batch_size=32, test_split=0.2):
        """Returns train and validation loaders from processed data."""
        x_path = os.path.join(self.processed_dir, 'X.npy')
        y_path = os.path.join(self.processed_dir, 'y.npy')

        if not os.path.exists(x_path):
            print("Processed data not found. Processing now...")
            self.process_data()

        X = np.load(x_path)
        y = np.load(y_path)
            
        split_idx = int(len(X) * (1 - test_split))
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]

        train_x, val_x = X[:split_idx], X[split_idx:]
        train_y, val_y = y[:split_idx], y[split_idx:]
        
        train_ds = MITBIHDataset(train_x, train_y)
        val_ds = MITBIHDataset(val_x, val_y)
        
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
               DataLoader(val_ds, batch_size=batch_size)

    def _process_records(self, record_names):
        """Extracts heartbeats from WFDB records."""
        all_X = []
        all_y = []
        
        # Mapping MIT-BIH labels to 5 AAMI classes (simplified)
        label_map = {
            'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Non-ectopic
            'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular ectopic
            'V': 2, 'E': 2,                          # Ventricular ectopic
            'F': 3,                                  # Fusion
            '/': 4, 'f': 4, 'Q': 4                   # Unknown / Paced
        }

        for name in tqdm(record_names, desc="Processing records"):
            record_path = os.path.join(self.raw_dir, name)
            try:
                record = wfdb.rdrecord(record_path)
                ann = wfdb.rdann(record_path, 'atr')
                
                signal = record.p_signal[:, 0] # Lead II usually
                for i, label in enumerate(ann.symbol):
                    if label in label_map:
                        peak = ann.sample[i]
                        if 128 <= peak < len(signal) - 128:
                            heartbeat = signal[peak-128 : peak+128]
                            # Normalize
                            heartbeat = (heartbeat - np.mean(heartbeat)) / (np.std(heartbeat) + 1e-8)
                            all_X.append(heartbeat)
                            all_y.append(label_map[label])
            except Exception as e:
                print(f"Error processing {name}: {e}")
                continue
                        
        return np.array(all_X), np.array(all_y)

class MITBIHDataset(Dataset):
    """PyTorch Dataset for MIT-BIH heartbeat classification."""
    
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data).unsqueeze(1) # [N, 1, seq_len]
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
