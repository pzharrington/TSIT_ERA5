import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
import torch.nn.functional as F

from models.pix2pix_model import Pix2PixModel
from models.afnonet import AFNONet, PrecipNet, load_afno
from utils.viz import *
from utils.YParams import *
from utils.spectra_metrics import *
from utils.weighted_acc_rmse import weighted_acc_torch_channels, weighted_rmse_torch_channels, \
    unlog_tp_torch, lat, latitude_weighting_factor_torch
from utils.precip_hists import precip_histc, precip_histc2, precip_histc3, binned_precip_log_l1
from utils.data_loader_multifiles import GetDataset

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from ruamel.yaml import YAML
from tqdm import tqdm


def setup_run_path(run, root_path='/pscratch/sd/j/jpduncan/weatherbenching/ERA5_generative'):
    """
    run: one of
        - str: absolute path to a run dir w/ "checkpoints" subdir and hyperparams.yaml file
        - str: path relative to root_path to a run dir w/ "checkpoints" subdir and hyperparams.yaml file
        - dict: run specification dictionary with "config_name" entry
        - dict: sweep specification dictionary with "sweep_id", "config_name", and "run_id" entries
    """
    if isinstance(run, dict):
        if 'run_path' in run.keys():
            return run['run_path']
        if 'root_path' in run.keys():
            root_path = run['root_path']
        if 'sweep_id' in run.keys():
            # old way of saving sweep runs used SLURM job id instead of run_id
            run_id = run['job_id'] if 'job_id' in run.keys() else run['run_id']
            if '/sweeps/' in root_path:
                # e.g., root_path '/pscratch/sd/j/jpduncan/tsitprecip/experiments/sweeps/weatherbenching/ERA5_generative'
                return os.path.join(root_path, f'{run["sweep_id"]}/{run["config_name"]}/{run_id}')
            else:
                return os.path.join(root_path, 'sweeps', f'{run["sweep_id"]}/{run["config_name"]}/{run_id}')

        else:
            return os.path.join(root_path, f'expts/{run["config_name"]}')
    elif os.path.isdir(run):
        # run is an absolute path
        run_path = run
    elif os.path.isdir(os.path.join(root_path, run)):
        # run is relative to root_path
        run_path = os.path.join(root_path, run)
    else:
        # run is a config name
        run_path = os.path.join(root_path, f'expts/{run}')

    return run_path


def setup_save_dir(run,
                   run_name: str = None,
                   use_best=True,
                   overwrite=False,
                   root_path='/global/cfs/cdirs/dasrepo/jpduncan/weatherbenching/ERA5_generative'):

    if isinstance(run, dict):
        if 'use_best' in run.keys():
            use_best = run['use_best']

        if 'sweep_id' in run.keys():
            assert run_name is not None or 'run_name' in run.keys(), \
                'please provide run_name with sweep'

        if 'run_name' in run.keys():
            save_subdir = run['run_name']
        elif 'config_name' in run.keys():
            save_subdir = run['config_name']
        elif 'run_id' in run.keys():
            save_subdir = run['run_id']
        else:
            assert run_name is not None
            save_subdir = run_name

        if 'overwrite' in run.keys():
            overwrite = run['overwrite']

    else:
        assert isinstance(run, str), 'run should be dict or str'
        if run_name is not None:
            save_subdir = run_name
        else:
            assert 'sweeps' not in run, \
                'please provide run_name with sweep'
            if 'expts' in run:
                save_subdir = re.search(r'(?<=expts/).*')[0]
            else:
                # run is a config name
                save_subdir = run

    if use_best:
        save_dir = os.path.join(root_path, save_subdir, 'ckpt_best')
    else:
        save_dir = os.path.join(root_path, save_subdir, 'ckpt')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        overwrite = True

    return save_dir, overwrite


def setup_params(run, train=False,
                 root_path='/pscratch/sd/j/jpduncan/weatherbenching/ERA5_generative',
                 base_config_path='/global/homes/j/jpduncan/intern/TSIT_ERA5/config/tsit.yaml'):
    """
    run: see setup_run_path
    """
    run_path = setup_run_path(run, root_path)

    # load params
    config_path = f'{run_path}/hyperparams.yaml'
    params = YParams(base_config_path, 'base')

    # convert hyperparams.yaml entries to correct types
    with open(config_path) as _file:
        for key, val in YAML().load(_file).items():
            if key in params.params.keys() and params[key] is not None:
                # print(f'{key}: {val}')
                if val[0] == '[':
                    val = [int(v) for v in val[1:-1].split(',')]
                if isinstance(params[key], bool):
                    val = val == 'True'
                params[key] = type(params[key])(val)

    return params


def setup_replicas(run, train=False, devs=[0], use_best=True,
                   root_path='/pscratch/sd/j/jpduncan/weatherbenching/ERA5_generative',
                   base_config_path='/global/homes/j/jpduncan/intern/TSIT_ERA5/config/tsit.yaml'):
    """
    run: see setup_run_path
    """
    run_path = setup_run_path(run, root_path)

    if isinstance(run, dict) and 'use_best' in run.keys():
        use_best = run['use_best']

    if use_best:
        ckpt_file = 'ckpt_best.tar'
    else:
        ckpt_file = 'ckpt.tar'

    ckpt_path = f'{run_path}/checkpoints/{ckpt_file}'

    params = setup_params(run, train, root_path, base_config_path)

    pix2pix_models = [None for dev in range(max(devs) + 1)]
    for dev in devs:
        # load model
        pix2pix_model = Pix2PixModel(params, distributed=False, local_rank=dev, device=dev, isTrain=train)
        checkpoint = torch.load(ckpt_path, map_location=f'cuda:{dev}')

        for key in checkpoint.keys():
            if 'model_state' in key and checkpoint[key] is not None:
                consume_prefix_in_state_dict_if_present(checkpoint[key], 'module.')

        pix2pix_model.load_state(checkpoint)
        pix2pix_model.set_eval()
        pix2pix_models[dev] = pix2pix_model

    return pix2pix_models


def setup_data(params,
               global_batch_size=1,
               num_workers=1,
               distributed=False,
               train=False,
               shuffle=True,
               devs=[0]):

    files_pattern = params.train_data_path if train else params.valid_data_path
    dataset = GetDataset(params, files_pattern, train)

    local_batch_size = global_batch_size // len(devs)

    # get data loaders
    data_loaders = [None for dev in range(max(devs) + 1)]
    tp_tms = [None for dev in range(max(devs) + 1)]
    for dev in devs:
        sampler = DistributedSampler(dataset,
                                     num_replicas=len(devs),
                                     rank=dev,
                                     shuffle=shuffle) if distributed else None

        data_loaders[dev] = DataLoader(dataset,
                                       batch_size=local_batch_size,
                                       num_workers=num_workers,
                                       shuffle=(shuffle and sampler is None),
                                       sampler=sampler,
                                       drop_last=True,
                                       pin_memory=True)

        # load tp time means
        tp_tms[dev] = torch.as_tensor(np.load(params.precip_time_means)).to(f'cuda:{dev}')

    return dataset, data_loaders, tp_tms


def setup_afno_wind(params, devs=[0]):
    afno_wind_models = [None for dev in range(max(devs) + 1)]
    for dev in devs:
        params.N_in_channels = params.afno_wind_N_channels
        params.N_out_channels = params.afno_wind_N_channels
        afno_wind = AFNONet(params).to(f'cuda:{dev}')
        afno_wind = load_afno(afno_wind,
                              params,
                              params.afno_model_wind_path,
                              map_location=torch.device(f'cuda:{dev}'))
        afno_wind_models[dev] = afno_wind.to(f'cuda:{dev}')
    return afno_wind_models


def setup_afno_precip(params, devs=[0]):
    params.N_in_channels = params.afno_wind_N_channels
    params.N_out_channels = len(params.out_channels)
    afno_precip_models = [None for dev in range(max(devs) + 1)]
    for dev in devs:
        afno_precip = AFNONet(params).to(dev)
        afno_precip = PrecipNet(params, backbone=afno_precip).to(dev)
        afno_precip = load_afno(afno_precip,
                                params,
                                params.afno_model_precip_path,
                                map_location=torch.device(f'cuda:{dev}'))
        afno_precip_models[dev] = afno_precip.to(f'cuda:{dev}')
    return afno_precip_models


def calc_inference_summaries(outputs: dict, tp_tm,
                             summarize_era5=True,
                             precip_eps=1e-5,
                             bins=300, bins_max=11.):
    """
    outputs: should have keys "pred" and "target", and optionally "pred_afno"

    modifies outputs
    """
    out = outputs
    gen = out['pred']
    tar = out.pop('era5')

    out['pred_spec'] = torch.fft.rfft(gen.float(), dim=-1, norm='ortho')[:, 0].abs().mean(dim=-2)

    gen_unlog = unlog_tp_torch(gen, precip_eps)
    tar_unlog = unlog_tp_torch(tar, precip_eps)
    out['pred_acc'] = weighted_acc_torch_channels(gen_unlog - tp_tm, tar_unlog - tp_tm)

    if 'afno' in out.keys():
        gen_afno = out['afno']
        gen_unlog = unlog_tp_torch(gen_afno, precip_eps)
        out['afno_spec'] = torch.fft.rfft(gen_afno.float(), dim=-1, norm='ortho')[:, 0].abs().mean(dim=-2)
        out['afno_acc'] = weighted_acc_torch_channels(gen_unlog - tp_tm, tar_unlog - tp_tm)
        pred_hists, tar_hists, afno_hists = precip_histc3(gen, tar, gen_afno, bins=bins, max=bins_max)
        out['afno_hists'] = afno_hists
        out['afno_binned_log_l1'] = binned_precip_log_l1(gen_afno, tar, afno_hists, tar_hists,
                                                         bins=bins, max=bins_max)
    else:
        pred_hists, tar_hists = precip_histc2(gen, tar, bins=bins, max=bins_max)

    out['pred_hists'] = pred_hists
    out['pred_binned_log_l1'] = binned_precip_log_l1(gen, tar, pred_hists, tar_hists,
                                                     bins=bins, max=bins_max)

    if summarize_era5:
        out['era5'] = tar
        out['era5_spec'] = torch.fft.rfft(tar.float(), dim=-1, norm='ortho')[:, 0].abs().mean(dim=-2)
        out['era5_hists'] = tar_hists

    return out


def gen_precip(data_loader, pix2pix_model, tp_tm,
               afno_wind_model=None,
               afno_precip_model=None,
               summarize_era5=True,
               bins=300, bins_max=11.,
               total_batches=None):
    """ Returns a Python generator which iterates over the data in data_loader
    and calculates metrics.

    """

    with torch.inference_mode(), torch.cuda.amp.autocast(True):

        for idx, (image_batch, target_batch) in enumerate(data_loader):

            if total_batches is None or idx < total_batches:

                dev = pix2pix_model.device
                params = pix2pix_model.params

                data = (image_batch.to(f'cuda:{dev}'), target_batch.to(f'cuda:{dev}'))

                if params.train_on_afno_wind or afno_precip_model is not None:
                    assert afno_wind_model is not None
                    n_wind = params['afno_wind_N_channels']
                    afno_wind = afno_wind_model(data[0][:, :n_wind]).detach()
                    if params.train_on_afno_wind:
                        if params.add_grid:
                            data = (torch.cat([afno_wind, data[0][:, n_wind:]], dim=1), data[1])
                        else:
                            data = (afno_wind, data[1])

                gen, _ = pix2pix_model.generate_fake(data[0], data[1], validation=True)

                out = {}
                out['pred'] = gen
                out['era5'] = data[1]

                if afno_precip_model is not None:
                    out['afno'] = afno_precip_model(afno_wind)

                out = calc_inference_summaries(out, tp_tm,
                                               summarize_era5=summarize_era5,
                                               precip_eps=params.precip_eps,
                                               bins=bins, bins_max=bins_max)
                yield out

            else:
                break


def get_precip_stream(run,
                      gen_afno_precip=True,
                      summarize_era5=True,
                      bins=300, bins_max=11.,
                      global_batch_size=12,
                      num_workers=4,
                      n_gpu=4,
                      total_batches=None):
    """
    run: see setup_run_path
    """

    torch.cuda.empty_cache()

    with torch.inference_mode(), torch.cuda.amp.autocast(True):

        pix2pix_models = setup_replicas(run, devs=np.arange(n_gpu))
        params = pix2pix_models[0].params

        valid_dataset, valid_data_loaders, tp_tms = setup_data(params,
                                                               global_batch_size=global_batch_size,
                                                               num_workers=num_workers,
                                                               distributed=True,
                                                               train=False,
                                                               devs=np.arange(n_gpu))

        if params['train_on_afno_wind'] or gen_afno_precip:
            afno_wind_models = setup_afno_wind(params, devs=np.arange(n_gpu))
        else:
            afno_wind_models = [None] * n_gpu

        if gen_afno_precip:
            afno_precip_models = setup_afno_precip(afno_wind_models[0].params, devs=np.arange(n_gpu))
        else:
            afno_precip_models = [None] * n_gpu

        valid_data_loaders[0] = tqdm(valid_data_loaders[0], total=total_batches)

        stream = zip(*[gen_precip(vdl,
                                  pix2pix_model=pix2pix_models[i],
                                  tp_tm=tp_tms[i],
                                  afno_wind_model=afno_wind_models[i],
                                  afno_precip_model=afno_precip_models[i],
                                  summarize_era5=summarize_era5,
                                  bins=bins, bins_max=bins_max,
                                  total_batches=total_batches)
                       for i, vdl in enumerate(valid_data_loaders)])

        return stream


def validation(runs,
               bins=300, bins_max=11.,
               global_batch_size=32,
               num_workers=16,
               n_gpu=4,
               total_batches=None,
               save=True,
               overwrite=False,
               save_root='/global/cfs/cdirs/dasrepo/jpduncan/weatherbenching/ERA5_generative'):

    summaries = {}

    gen_afno_precip = True
    summarize_era5 = True

    afno_fpath = os.path.join(save_root, 'afno', 'validation.pkl')
    era5_fpath = os.path.join(save_root, 'era5', 'validation.pkl')

    # load afno summary
    if os.path.isfile(afno_fpath) and not overwrite:
        with open(afno_fpath, 'rb') as f:
            summaries['afno'] = pickle.load(f)
        gen_afno_precip = False
    elif save and not os.path.isdir(os.path.join(save_root, 'afno')):
        os.makedirs(os.path.join(save_root, 'afno'))

    # load era5 summary
    if os.path.isfile(era5_fpath) and not overwrite:
        with open(era5_fpath, 'rb') as f:
            summaries['era5'] = pickle.load(f)
        summarize_era5 = False
    elif save and not os.path.isdir(os.path.join(save_root, 'era5')):
        os.makedirs(os.path.join(save_root, 'era5'))

    bin_edges = np.linspace(0., bins_max, bins + 1)

    for name, run in runs.items():

        save_dir, overwrite_ = setup_save_dir(run,
                                              overwrite=overwrite,
                                              root_path=save_root)

        save_fpath = os.path.join(save_dir, 'validation.pkl')

        if overwrite_ or not os.path.isfile(save_fpath):
            print(f'validating "{name}" run')
            stream = get_precip_stream(run,
                                       gen_afno_precip=gen_afno_precip,
                                       summarize_era5=summarize_era5,
                                       bins=bins, bins_max=bins_max,
                                       global_batch_size=global_batch_size,
                                       num_workers=num_workers,
                                       n_gpu=n_gpu,
                                       total_batches=total_batches)

            new_summary_keys = [name]
            if gen_afno_precip:
                new_summary_keys.append('afno')
            if summarize_era5:
                new_summary_keys.append('era5')

            # drops[i] is an output from gen_precip()
            for drops in stream:

                for i in range(n_gpu):

                    for key, output in drops[i].items():

                        if 'era5' in key:
                            inner_dict = summaries.setdefault('era5', {})
                        elif 'afno' in key:
                            inner_dict = summaries.setdefault('afno', {})
                        else:
                            inner_dict = summaries.setdefault(name, {})

                        if key in ['pred', 'era5', 'afno']:
                            new_key = 'fields'
                        else:
                            new_key = key[5:]

                        output_list = inner_dict.setdefault(new_key, [])
                        if not new_key == 'fields' or len(output_list) < n_gpu:
                            output_list.append(output)

            # move summaries to CPU and cat
            for key in new_summary_keys:
                new_summaries = summaries[key]
                for summary_name, summary in new_summaries.items():
                    new_summaries[summary_name] = torch.cat([x.to('cpu') for x in summary], dim=0).numpy()

            if save:
                with open(save_fpath, 'wb') as f:
                    pickle.dump(summaries[name], f)

                if 'afno' in new_summary_keys:
                    with open(os.path.join(afno_fpath), 'wb') as f:
                        pickle.dump(summaries['afno'], f)

                if 'era5' in new_summary_keys:
                    with open(os.path.join(era5_fpath), 'wb') as f:
                        pickle.dump(summaries['era5'], f)

        else:
            print(f'loading previously saved summary for "{name}" run')
            # load previously saved summary
            with open(save_fpath, 'rb') as f:
                summaries[name] = pickle.load(f)

        # only need to do these once
        summarize_era5 = not 'era5' in summaries.keys()
        gen_afno_precip = not 'afno' in summaries.keys()

    return summaries, bin_edges
