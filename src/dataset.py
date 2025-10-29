# PyTorch Dataset/DataLoader for client data

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from src.config import BATCH_SIZE

class ClientDataset(Dataset):
    """PyTorch Dataset for loading a single client's .npz data."""
    def __init__(self, data_path):
        data = np.load(data_path)
        self.X = torch.tensor(data['X'], dtype=torch.float32)
        self.y = torch.tensor(data['y'], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloader(data_path):
    """Returns a DataLoader for a client's dataset."""
    dataset = ClientDataset(data_path)
    # Split 80% train, 20% validation
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
    
    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    return trainloader, valloader, len(train_set)