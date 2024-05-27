import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import xarray as xr
import pandas as pd
import torch


class DatasetERA5andMeasurements(Dataset):
    """
    Dataset containg the ERA5 analysis (X) and the surface flux measurements (Y)

    """    
    def __init__(self,X=None,Y=None, var_list=["SSHF", "SLHF", "SSRD", "SSR", "STRD", "STR", "T2M", "U10M", "V10M", "SKT"]):
        """
        Initialise dataset, read ERA5
        """
        super().__init__()
        self.var_list = var_list
        
        # READ ERA5 OUTPUT
        ds_list = []
        # READ ERA5 ALONG MOSAIC PATH
        for var in var_list:
            fname = f"/work/ab1385/a270164/2024-sebai/ERA5/E5sf121H_201910_202009_{var}_mc.nc"
            try:
                ERA5_data = xr.open_dataset(fname)
                ds_list.append(ERA5_data)
            except Exception as e:
                print(f"Error loading data from {fname}: {e}")
    
        if not ds_list:
            print("No data available for the specified variables.")
            
        self.ds = xr.merge(ds_list)      # ds should have dimension time and multiple variables   

        # READ MOSAIC SURFACE FLUXES - Sensible heat flux
        surface_fluxes = xr.open_dataset("/work/ab1385/a270164/2024-sebai/MOSAiC/mosseb_metcity_34_hr.nc")
        surface_fluxes_var = surface_fluxes['Hs_2m']       

        # Remove the 'time_bnds' coordinate from surface_fluxes
        surface_fluxes = surface_fluxes.drop_vars('time_bnds')
        
        self.Y_np = surface_fluxes_var.values
        
    def normalize(self, datatonorm):
        """
        Normalize data using mean and standard deviation (Z-score is a variation of scaling that represents the number 
        of standard deviations away from the mean. You would use z-score to ensure your feature distributions have mean = 0 and std = 1. )
        """
        mean = np.nanmean(datatonorm, axis=0)
        std = np.nanstd(datatonorm, axis=0)
        normalized_data = (datatonorm - mean) / std
        return normalized_data, mean, std
    
    def denormalize(self, normalized_data, mean, std):
        """
        Denormalize data using mean and standard deviation
        """
        return normalized_data * std + mean

    def to_torch(self):        
        # Extract variable values from xarray dataset
        var_values = [self.ds[var].values for var in self.var_list]
    
        # Concatenate variable values along the last axis
        self.X_np = np.stack(var_values, axis=-1).astype(np.float64)

        # remove nan values from the both X_np and Y_np
        self.X_np = self.X_np[~np.isnan(self.Y_np)]
        self.Y_np = self.Y_np[~np.isnan(self.Y_np)]
        
        # Store the valid time indices
        self.valid_time_indices = np.where(~np.isnan(self.Y_np))[0]       
        
        # Normalize X and Y data
        self.X_np, self.X_mean, self.X_std = self.normalize(self.X_np)
        self.Y_np, self.Y_mean, self.Y_std = self.normalize(self.Y_np)
        
        # Convert X_np and Y_np to torch tensor
        self.X = torch.from_numpy(self.X_np)
        self.Y = torch.from_numpy(self.Y_np)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.X)


class SimpleNeuralNet(nn.Module):
    def __init__(self, input_size, hl = [64, 64, 64, 64, 64]):
        super().__init__()
        self.fc1 = nn.Linear(10, hl[0])
        self.fc2 = nn.Linear(hl[0], hl[1])
        self.fc3 = nn.Linear(hl[1], hl[2])
        self.fc4 = nn.Linear(hl[2], hl[3])
        self.fc5 = nn.Linear(hl[3], hl[4])
        self.fc6 = nn.Linear(hl[4], 1)
        self.double()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x
