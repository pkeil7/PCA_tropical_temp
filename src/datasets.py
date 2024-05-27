# Pytorch Dataset class
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import pandas as pd
import torch


class SimpleDataset(Dataset):

    """
    Dataset containg the ERA5 near-surface variables (X) and upper trop PCas (Y)

    """

    def __init__(self,X :np.ndarray, Y:np.ndarray):
        """

        Args:
            X (np.ndarray): near-surface vars
            Y (np.ndarray): upper tropospheric temperature PCAs
        """
        super().__init__()
        self.X = torch.from_numpy(X).to(torch.float32)
        self.Y = torch.from_numpy(Y).to(torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.X)
    
