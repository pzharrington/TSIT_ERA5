""" 
    misc utilties
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class PeriodicPad2d(nn.Module):
    """ 
        pad longitudinal (left-right) circular 
        and pad latitude (top-bottom) with zeros
    """
    def __init__(self, pad_width):
       super(PeriodicPad2d, self).__init__()
       self.pad_width = pad_width

    def forward(self, x):
        # pad left and right circular
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular") 
        # pad top and bottom zeros
        out = F.pad(out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0) 
        return out


def create_lat_mse(weights_lat):
    def lat_mse(y_pred, y_true):
        error = y_pred - y_true
        mse = error**2 * weights_lat[None,None,:,None]
        mse = torch.mean(mse)
        return mse
    return lat_mse

def compute_lat_rmse(pred, true, weights_lat):
    error = pred - true
    mse = error**2 * weights_lat[None,None,:,None]
    # (b,c,h,w) = mse.shape
    # compute error in the b,h,w dim for each channel
    rmse = torch.sqrt(torch.mean(mse, dim=[0,2,3]))
    return rmse

def unstandardize(x, mean, std):
    return (x * std[None,:,None,None]) + mean[None,:,None,None]

#def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
#    """
#    from weatherbench: https://github.com/pangeo-data/WeatherBench/blob/master/src/score.py
#    Compute the RMSE with latitude weighting from two xr.DataArrays.
#    Args:
#        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
#        da_true (xr.DataArray): Truth.
#        mean_dims: dimensions over which to average score
#    Returns:
#        rmse: Latitude weighted root mean squared error
#    """
#    error = da_fc - da_true
#    weights_lat = np.cos(np.deg2rad(error.lat))
#    weights_lat /= weights_lat.mean()
#    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
#    return rmse

def log_trans(x, e):
    return np.log(x + e) - np.log(e)

def log_retrans(x, e):
    return np.exp(x + np.log(e)) - e


def viz_fields(fields):
    pred, tar = fields
    fig = plt.figure(figsize=(12,12))
    for i, tag in enumerate(['z500', 't850', 't2m']):
        n = np.abs(tar[i,:,:]).max()
        plt.subplot(3, 3, i+1)
        plt.title('%s, pred'%tag)
        field = pred[i]
        plt.imshow(np.roll(field[::-1,:], shift=field.shape[1]//2, axis=1), cmap = 'coolwarm', norm=Normalize(-n,n))
        plt.axis('off')
        plt.subplot(3, 3, i+4)
        plt.title('%s, true'%tag)
        field = tar[i]
        plt.imshow(np.roll(field[::-1,:], shift=field.shape[1]//2, axis=1), cmap = 'coolwarm', norm=Normalize(-n,n))
        plt.axis('off')
        plt.subplot(3, 3, i+7)
        plt.title('%s, residual'%tag)
        field = pred[i] - tar[i]
        plt.imshow(np.roll(field[::-1,:], shift=field.shape[1]//2, axis=1), cmap = 'seismic', norm=Normalize(-n,n))
        plt.axis('off')
    plt.tight_layout()
    return fig



