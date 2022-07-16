import logging
import glob
from types import new_class
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py
import math
import torchvision.transforms.functional as TF
import matplotlib
import matplotlib.pyplot as plt

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

def reshape_fields(img, inp_or_tar, crop_size_x, crop_size_y,rnd_x, rnd_y, params, y_roll, train, normalize=True):
    #Takes in np array of size (n_history+1, c, h, w) and returns torch tensor of size ((n_channels*(n_history+1), crop_size_x, crop_size_y)

    if len(np.shape(img)) ==3:
      img = np.expand_dims(img, 0)

    img = img[:, :, 0:720] #remove last pixel
    n_history = np.shape(img)[0] - 1
    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]
    n_channels = np.shape(img)[1] #this will either be N_in_channels or N_out_channels
    n_grid_channels = 0 # updated when creating the grid
    in_channels = np.arange(params.afno_wind_N_channels) if not train and params.afno_validate else params.in_channels
    channels = in_channels if inp_or_tar =='inp' else params.out_channels
    mins = np.load(params.min_path)[:, channels]
    maxs = np.load(params.max_path)[:, channels]
    means = np.load(params.global_means_path)[:, channels]
    stds = np.load(params.global_stds_path)[:, channels]
    if crop_size_x == None:
        crop_size_x = img_shape_x
    if crop_size_y == None:
        crop_size_y = img_shape_y

    if normalize:
        if params.normalization == 'minmax':
          img  -= mins
          img /= (maxs - mins)
        elif params.normalization == 'zscore':
          img -=means
          img /=stds

    if params.add_grid and inp_or_tar == 'inp':
        n_grid_channels = params.N_grid_channels
        if params.gridtype == 'linear':
            x = np.meshgrid(np.linspace(-1, 1, img_shape_x))
            y = np.meshgrid(np.linspace(-1, 1, img_shape_y))
            grid_x, grid_y = np.meshgrid(y, x)
            if params.gridalong == 'x':
                grid = np.expand_dims(grid_y, axis=[0, 1])
            elif params.gridalong == 'y':
                grid = np.expand_dims(grid_x, axis=[0, 1])
            else:
                grid = np.expand_dims(np.stack((grid_x, grid_y), axis = 0), axis=0)
        elif params.gridtype == 'sinusoidal':
            x1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_x)))
            x2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_x)))
            y1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_y)))
            y2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_y)))
            grid_x1, grid_y1 = np.meshgrid(y1, x1)
            grid_x2, grid_y2 = np.meshgrid(y2, x2)
            if params.gridalong == 'x':
                grid = np.expand_dims(np.stack((grid_y1, grid_y2), axis = 0), axis = 0)
            elif params.gridalong == 'y':
                grid = np.expand_dims(np.stack((grid_x1, grid_x2), axis = 0), axis = 0)
            else:
                grid = np.expand_dims(np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis = 0), axis = 0)
        grid = np.resize(grid, (img.shape[0], ) + grid.shape[1:]).astype(np.float32)
        assert n_grid_channels == grid.shape[1], \
            f'N_grid_channels must be set to {grid.shape[1]} for {params.gridtype} grid along axes "{params.gridalong}"'
        img = np.concatenate((img, grid), axis = 1 )

    if params.roll:
        img = np.roll(img, y_roll, axis = -1)

    if train and (crop_size_x or crop_size_y):
        img = img[:,:,rnd_x:rnd_x+crop_size_x, rnd_y:rnd_y+crop_size_y]

    if inp_or_tar == 'inp':
        img = np.reshape(img, (n_channels*(n_history+1) + n_grid_channels, crop_size_x, crop_size_y))
    elif inp_or_tar == 'tar':
        img = np.reshape(img, (n_channels + n_grid_channels, crop_size_x, crop_size_y))

    img = torch.as_tensor(img)

    if params.img_size != img.shape:
        outx, outy = params.img_size
        img = TF.resize(img, size=(outx,outy))
    return img

def reshape_precip(img, inp_or_tar, crop_size_x, crop_size_y,rnd_x, rnd_y, params, y_roll, train, normalize=True):

    if len(np.shape(img)) ==2:
      img = np.expand_dims(img, 0)

    img = img[:,:720,:]
    img_shape_x = img.shape[-2]
    img_shape_y = img.shape[-1]
    n_channels = 1
    n_grid_channels = 0 # updated when creating the grid
    if crop_size_x == None:
        crop_size_x = img_shape_x
    if crop_size_y == None:
        crop_size_y = img_shape_y

    if normalize:
        eps = params.precip_eps
        img = np.log1p(img/eps)

    if params.add_grid and inp_or_tar == 'inp':
        n_grid_channels = params.N_grid_channels if params.add_grid else 0
        if params.gridtype == 'linear':
            x = np.meshgrid(np.linspace(-1, 1, img_shape_x))
            y = np.meshgrid(np.linspace(-1, 1, img_shape_y))
            grid_x, grid_y = np.meshgrid(y, x)
            if params.gridalong == 'x':
                grid = np.expand_dims(grid_y, axis=[0, 1])
            elif params.gridalong == 'y':
                grid = np.expand_dims(grid_x, axis=[0, 1])
            else:
                grid = np.expand_dims(np.stack((grid_x, grid_y), axis = 0), axis=0)
        elif params.gridtype == 'sinusoidal':
            x1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_x)))
            x2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_x)))
            y1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_y)))
            y2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_y)))
            grid_x1, grid_y1 = np.meshgrid(y1, x1)
            grid_x2, grid_y2 = np.meshgrid(y2, x2)
            if params.gridalong == 'x':
                grid = np.expand_dims(np.stack((grid_y1, grid_y2), axis = 0), axis = 0)
            elif params.gridalong == 'y':
                grid = np.expand_dims(np.stack((grid_x1, grid_x2), axis = 0), axis = 0)
            else:
                grid = np.expand_dims(np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis = 0), axis = 0)
        grid = np.resize(grid, (img.shape[0], ) + grid.shape[1:]).astype(np.float32)
        assert n_grid_channels == grid.shape[1], \
            f'N_grid_channels must be set to {grid.shape[1]} for {params.gridtype} grid along axes "{params.gridalong}"'
        img = np.concatenate((img, grid), axis = 1 )

    if params.roll:
        img = np.roll(img, y_roll, axis = -1)

    if train and (crop_size_x or crop_size_y):
        img = img[:,rnd_x:rnd_x+crop_size_x, rnd_y:rnd_y+crop_size_y]

    img = torch.as_tensor(np.reshape(img, (n_channels + n_grid_channels, crop_size_x, crop_size_y)))

    if params.img_size != img.shape:
        outx, outy = params.img_size
        img = TF.resize(img, size=(outx,outy))

    return img

def compute_latent_vector_size(params):
    num_blocks = params.num_upsampling_blocks

    img_size_log2 = 2**int(np.log2(params.img_size[0])), \
        2**int(np.log2(params.img_size[1]))

    if params.DEBUG:
        assert img_size_log2[0] == 512 and img_size_log2[1] == 1024, \
            f'Unexpected img_size_log2: {img_size_log2}'

    if img_size_log2 != params.img_size:
        sw, sh = img_size_log2[0] // (2**(num_blocks-1)), \
            img_size_log2[1] // (2**(num_blocks-1))

        if params.DEBUG and num_blocks == 8:
            assert sw == 4 and sh == 8, f'Unexpected (sw, sh): {(sw, sh)}'

        if params.DEBUG and num_blocks == 7:
            assert sw == 8 and sh == 16, f'Unexpected (sw, sh): {(sw, sh)}'

    else:
        sw, sh = img_size_log2[0] // (2**num_blocks), \
            img_size_log2[1] // (2**num_blocks)

    return sw, sh, num_blocks
