"""
  data loaders
"""
import re
import time
import os, sys
import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import h5py
from utils.variable_codes import *
from utils.misc_utils import *

def load_data(data_dir, var_dict, years):
    ds = xr.merge(
        [xr.open_mfdataset(os.path.join(*[data_dir, var, "*.nc"]), combine='by_coords')
            for var in var_dict.keys()],
        fill_value=0  # for nans
        )
    return ds.sel(time=slice(*years))

#def get_data_loader(params, data_dir, distributed, var_dict, lead_time=6, mean=None, std=None, data_split=0):
def get_data_loader(params, data_file, constants_file, stats_file, distributed, var_dict, lead_time=6, data_split=0):
    """ 
        data_split = 0 for train, 1 for valid, 2 for test
    """
    transform = torch.from_numpy
    #dataset = WeatherBenchDataset(params, data_dir, transform, var_dict, lead_time, None, None, data_split)
    dataset = WeatherBenchDatasetH5(params, data_file, constants_file, stats_file, transform, var_dict, lead_time, data_split)
    sampler = DistributedSampler(dataset, shuffle=(data_split == 0)) if distributed else None
    dataloader = DataLoader(dataset,
                            batch_size=int(params.batch_size) if data_split==0 else int(params.valid_batch_size_per_gpu),
                            num_workers=params.num_data_workers,
                            shuffle=(sampler is None and (data_split == 0)),
                            sampler=sampler,
                            drop_last=False,#(data_split==0),
                            pin_memory=torch.cuda.is_available())

#    return dataloader, sampler, dataset.mean, dataset.std
    return dataloader #, sampler


class WeatherBenchDatasetH5(Dataset):
    """ 
        Dataset loading assuming a single hdf5 file for full data
    """
    def __init__(self, params, fname, constants_file, stats_file, transform, var_dict, lead_time, data_split):
        self.params = params
        self.fname = fname
        self.fname_const = constants_file
        self.transform = transform
        self.lead_time = lead_time
        self.var_dict = var_dict
        self.tp_log = 0.001
        self.dt = 1
        self.dt_in = params.dt_in # spacing btw time steps for inputs in hours
        self.nt_in = params.nt_in # number of input times
        self.dt_in = int(self.dt_in // self.dt)
        self.nt_offset = (self.nt_in - 1) * self.dt_in
        self.lead_time = int(self.lead_time / self.dt)
        self.level_names = []

        self.n_channels = 0
        for nm, var_params in var_dict.items():
            if nm == "constants": # these are constant fields like latitude
                for var in var_params:
                    self.level_names.append(var)
                    self.n_channels += 1
            else:
                var, levels = var_params
                if levels is not None:
                    self.level_names += [f'{var}_{level}' for level in levels]
                    self.n_channels += self.nt_in*len(levels)
                else:
                    self.level_names.append(var)
                    self.n_channels += self.nt_in

        self.output_vars = output_vars if params.outputs == 'zzt' else output_vars_tp
        self.output_idxs = [i for i, l in enumerate(self.level_names)
                                if any([bool(re.match(o, l)) for o in self.output_vars])]
        self.n_targets = len(self.output_idxs)
        if params.log_to_screen:
            logging.info("using {} output variables".format(self.n_targets))
        self.const_idxs = [i for i, l in enumerate(self.level_names) if l in var_dict['constants']]
        if self.params.io_match:
            self.not_const_idxs = [i for i, l in enumerate(self.level_names) if l in self.output_vars]
            self.n_channels = len(self.not_const_idxs) * self.nt_in + len(self.const_idxs)
        else:
            self.not_const_idxs = [i for i, l in enumerate(self.level_names) if l not in var_dict['constants']]

        self.coords = {}
        with h5py.File(stats_file, 'r') as f:
            self.mean = f["era5_mean"][:] 
            self.std = f["era5_std"][:] 
            self.coords['lat'] = f["era5_lat"][:]
            self.coords['lon'] = f["era5_lon"][:]

        if data_split == 0:
            self.clip = 24 # clip the first 24 hrs for training data
        else:
            self.clip = 0

        self.n_samples, _, self.ih, self.iw = self.compute_len()
        self.n_samples -= self.clip
        if not params.use_constant_inputs:
            self.n_channels -= len(self.const_idxs)

    def compute_len(self):
        with h5py.File(self.fname, 'r') as f:
            sh = f['era5_fields'].shape
            return sh[0] - self.nt_offset - self.lead_time, sh[1], sh[2], sh[3]

    def _open_file(self, path):
        return h5py.File(path, 'r')

    def __len__(self):
        return self.n_samples

    def get_data(self, dat, idxs=None):
        ''' standardize the data according to level '''
        sh = dat.shape
        len_sh = len(sh)
        if len_sh == 3:
            lvls = sh[0]
        else:
            lvls = sh[1]

        if not idxs:
            idxs = [*range(0,lvls)] # all lvls
        mean = self.mean[idxs]
        std = self.std[idxs]

        if len_sh == 3:
            # only one time point or constants
            dat_stdized = ((dat - mean[:,None,None]) / std[:,None,None])
        else:
            dat_stdized = ((dat - mean[None,:,None,None]) / std[None,:,None,None])
            dat_stdized = np.reshape(dat_stdized, (sh[0]*sh[1], sh[2], sh[3])) # reshape the times into channels
        return dat_stdized

    def __getitem__(self, idx):
        if not hasattr(self, 'dat'):
            self.dat = self._open_file(self.fname)
            self.dat_const = self._open_file(self.fname_const)
        dat = self.dat
        dat_const = self.dat_const
        start = self.nt_offset + idx + self.clip
        if not self.params.io_match:
            time_idxs = [*range(start - (self.nt_in-1)*self.dt_in, start + self.dt_in, self.dt_in)]
            X = [self.get_data(dat['era5_fields'][time_idxs,:,:,:])]
        else:
            X = [self.get_data(dat['era5_fields'][start-pre,self.not_const_idxs,:,:], self.not_const_idxs) for pre in range((self.nt_in-1)*self.dt_in, -self.dt_in, -self.dt_in)]
        if self.params.use_constant_inputs:
            X.append(self.get_data(dat_const['era5_constants'][:,:,:], self.const_idxs)) # constants are (3,ih,iw)
        X = np.concatenate(X).astype('float32') # concat all input timesteps along channel dim
        y = self.get_data(dat['era5_fields'][start+self.lead_time,self.output_idxs,:,:], self.output_idxs).astype('float32') # targets
        return self.transform(X.transpose((0,2,1))), self.transform(y.transpose((0,2,1)))

