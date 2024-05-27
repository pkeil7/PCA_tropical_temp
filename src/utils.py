import xarray as xr
import numpy as np
import aes_thermo as mt
import torch
import cartopy.crs as ccrs


def normalise(ds,dim="time") :
    ds_mean = ds.mean(dim)
    ds_std = ds.std(dim)
    return (ds - ds_mean)/ds_std, ds_mean, ds_std

def RMSE(x,y) :
    return np.sqrt(np.mean((x-y)**2))


def xr_MSE(T,q):
    '''Takes temperature and specific humidtiy on pressure levels
    and returns moist static energy'''

    assert T.plev2.values.all() == q.plev2.values.all()

    z = get_z_from_p(T.plev2 *100.,270.)

    hd = mt.cpd*(T-mt.T0) + mt.gravity*z
    hv = mt.cpv*(T-mt.T0) + mt.gravity*z + mt.lv0

    h = hd*(1-q) + q*hv

    return h

def get_z_from_p(P,T):
    '''P and T are single values or vectors'''
    z = -mt.Rd *T / mt.gravity*(np.log(P)-np.log(101000.))
    return z


def trainer(epochs, train_dataloader, val_dataloader, model, optimizer, loss_function):
    """ 
    Train the model 
    
    """
    losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    for epoch in range(epochs):
        losses_epoch = []
        for x,y in train_dataloader:

            optimizer.zero_grad()
            # Make a prediction
            prediction = model(x)
            # Calculate the loss
            loss = loss_function(prediction.squeeze(), y)
            
            # the gradient is computed and stored.
            # .step() performs parameter update
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            losses_epoch.append(loss.detach().numpy())

        # Mean Loss for this epoch:
        losses[epoch] = np.array(losses_epoch).mean()   # np.mean(losses_epoch)       
        # validation loss for this epoch:
        val_losses[epoch] = validate(val_dataloader, model, loss_function)

        if epoch % 10 == 0 :
            print(f"Epochs: {epoch}/{epochs}")
            print(f"Training Loss : {losses[epoch]}")
            print(f"Validation Loss : {val_losses[epoch]}")
    
    return losses, val_losses


def validate(train_dataloader, model, loss_function):
    """ 
    Validate the model 
    
    """
    losses = []
    model.eval()
    for x,y in train_dataloader :
        # switch off backpropagation:
        with torch.no_grad() :
            prediction = model(x)
            # Calculate the loss
            losses.append(loss_function(prediction.squeeze(), y).detach().numpy())
    model.train()
    return np.mean(losses)

def remove_axes(ax) :
    spines_to_keep = ['left', 'bottom']
    for spine in spines_to_keep:
        ax.spines[spine].set_linewidth(0.5)
    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

def set_regional_cartopy(ax,projection=ccrs.PlateCarree(central_longitude=180),extent=[0, 359, -20, 20]) :
    ax.coastlines()
    ax.set_extent(extent, projection)