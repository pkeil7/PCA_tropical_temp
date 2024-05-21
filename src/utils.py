import xarray as xr
import numpy as np

def normalise(ds,dim="time") :
    ds_mean = ds.mean(dim)
    ds_std = ds.std(dim)
    return (ds - ds_mean)/ds_std, ds_mean, ds_std

def RMSE(x,y) :
    return np.sqrt(np.mean((x-y)**2))