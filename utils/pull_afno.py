import sys
sys.path.append('/global/homes/p/pharring/research/fno-era5/ERA5_wind')
import os
import numpy as np

import torch
from collections import OrderedDict
from networks.afnonet import AFNONet, PrecipNet
from utils.data_precip import get_data_loader
import torch.distributed as dist


def load_model(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint = torch.load(checkpoint_file)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def setup_afno(params):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    _, valid_dataset = get_data_loader(params, params.valid_data_path, dist.is_initialized(), train=False)
    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y

    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.in_channels) # same as in for the wind model
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels
    params.means = np.load(params.global_means_path)[0, out_channels] # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[0, out_channels]
    # load the models
    # load wind model
    if params.nettype_wind == 'afno':
        model_wind = AFNONet(params, precip=False).to(device)
    checkpoint_file  =  params['model_wind_path']
    model_wind = load_model(model_wind, params, checkpoint_file)
    model_wind = model_wind.to(device) 
    return model_wind, valid_dataset

