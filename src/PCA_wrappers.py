import xarray as xr
import numpy as np
from sklearn.decomposition import PCA, FastICA

def PCA_wrapper(ds,number_of_PCs = 3, var="t",level=300m, coord_name="coord") :
    """PCA wrapper for xarray

    Args:
        ds (xarray.Dataset): xarray dataset with 2 coordinates: first is usually time, second is lon, lat stacked
        number_of_PCs (int, optional): number of principal components to calculate. Defaults to 3.
        var (str, optional): varaibles name. Defaults to "t".
        level (int, optional): vertical level in hPa. Defaults to 300.
        coord_name (string, optional) : name of geographical (stacked) coordinate

    Returns:
        _type_: _description_
    """
    explained_variance = np.zeros(number_of_PCs)
    da = ds.fillna(0).sel(isobaricInhPa=level).t.load()

    principal= PCA(n_components=number_of_PCs)
    principal.fit(da)
    time_series = principal.transform(da)
    results = xr.DataArray(principal.components_, coords = {"PCAs": np.arange(number_of_PCs) +1, coord_name : da[coord_name] }).to_dataset(name="components")
    explained_variance= principal.explained_variance_ratio_
        
    da = xr.DataArray(time_series, coords = {"time" : ds.time, "PCAs": np.arange(number_of_PCs) +1 })
    results["time_series"] = da
    results["explained_variance"] = (["PCAs"], explained_variance )
    return results

def ICA_wrapper(ds,number_of_PCs = 3, var="ta",level=300) :
    """This is still unfinished!

    Args:
        ds (_type_): _description_
        number_of_PCs (int, optional): _description_. Defaults to 3.
        var (str, optional): _description_. Defaults to "ta".
        level (int, optional): _description_. Defaults to 300.

    Returns:
        _type_: _description_
    """
    time_series = np.zeros((len(ds.model_id), len(ds.time) , number_of_PCs))
    explained_variance = np.zeros((len(ds.model_id),  number_of_PCs))
    da_comp_list =[]
    ds_stacked = ds[var].stack(coord=["lon","lat"]).fillna(0).sel(plev2=level)
    for i,model in enumerate(ds_stacked.model_id) :
        da = ds_stacked.sel(model_id = model)
        principal= FastICA(n_components=number_of_PCs, max_iter=1000)
        principal.fit(da)
        x = principal.transform(da)
        da_comp_list.append(xr.DataArray(principal.components_, coords = {"PCAs": np.arange(number_of_PCs) +1, "coord" : ds_stacked.coord}).unstack("coord"))
        da_comp_list[-1]["model_id"] = model
        time_series[i,:,:] = x
        
    dataset = xr.concat(da_comp_list, "model_id").to_dataset(name="components")
    da = xr.DataArray(time_series, coords = {"model_id" : dataset.model_id, "time" : ds_stacked.time, "PCAs": np.arange(number_of_PCs) +1 })
    dataset["time_series"] = da
    return dataset
        

        