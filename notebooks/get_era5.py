import xarray as xr
import numpy as np
from cdo import *
cdo = Cdo()

type="sf" # surface ("sf") or pressure levels ("pl")
var="034" # 130 is temperature, 133 humidity, 034 is SST
directory = f"/pool/data/ERA5/E5/{type}/an/1M/{var}/"
target = "/work/ka1176/paul/PCA_tropical_temp/data/era5/"
interpolate="global_5"

years =np.arange(1940,2023)

for year in years :
    ds = xr.open_dataset(directory + f"E5{type}00_1M_{year}_{var}.grb", engine="cfgrib", indexpath='')
    #ds = ds.sel(isobaricInhPa=[1000,925,850,700])
    ds.to_netcdf(target + f"E5{type}00_1M_{year}_{var}.nc")
    cdo.remapdis(interpolate, input = target + f"E5{type}00_1M_{year}_{var}.nc", output = target + f"E5{type}00_1M_{year}_{var}_{interpolate}.nc", )
    