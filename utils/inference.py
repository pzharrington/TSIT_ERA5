import os
import re
import glob
import h5py
import time
import pickle
import logging
import numpy as np

from models.pix2pix_model import Pix2PixModel
from models.afnonet import AFNONet, PrecipNet, load_afno
from utils.viz import *
from utils.YParams import *
from utils.spectra_metrics import *
from utils.weighted_acc_rmse import weighted_acc_torch_channels, weighted_rmse_torch_channels, \
    unlog_tp_torch, top_quantiles_error_torch
from utils.precip_hists import precip_histc2, precip_histc3, binned_precip_log_l1
from utils.data_loader_multifiles import GetDataset
from utils.img_utils import reshape_fields, reshape_precip

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torchvision.transforms.functional import gaussian_blur

from ruamel.yaml import YAML
from tqdm import tqdm

DECORRELATION_TIME = 8 # 2 days for preicp


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


def setup_params(run,
                 root_path='/pscratch/sd/j/jpduncan/weatherbenching/ERA5_generative',
                 base_params=None):
    """
    run: see setup_run_path
    """
    run_path = setup_run_path(run, root_path)

    # load params
    config_path = f'{run_path}/hyperparams.yaml'
    if base_params is None:
        params = YParams('../config/tsit.yaml', 'base')
    else:
        params = base_params

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


def setup_replicas(run,
                   train=False,
                   devs=[0],
                   use_best_acc=True,
                   use_best_binned_log_l1=False,
                   root_path='/pscratch/sd/j/jpduncan/weatherbenching/ERA5_generative',
                   base_params=None):
    """
    run: see setup_run_path
    """

    if isinstance(run, dict):

        if 'use_best_acc' in run.keys():
            use_best_acc = run['use_best_acc']
            use_best_binned_log_l1 = False

        if 'use_best_binned_log_l1' in run.keys():
            use_best_binned_log_l1 = run['use_best_binned_log_l1']
            use_best_acc = False

    assert not (use_best_acc and use_best_binned_log_l1), \
        'use_best or use_best_binned_log_l1, or neither, but not both'

    params = setup_params(run, root_path, base_params)

    if use_best_acc:
        ckpt_path = 'ckpt_best.tar'
    elif use_best_binned_log_l1:
        ckpt_path = 'ckpt_best_binned_log_l1.tar'
    else:
        ckpt_path = 'ckpt.tar'

    run_path = setup_run_path(run, root_path)

    ckpt_path = f'{run_path}/checkpoints/{ckpt_path}'

    pix2pix_models = [None for _ in range(max(devs) + 1)]
    for dev in devs:
        # load model
        pix2pix_model = Pix2PixModel(params, distributed=False, local_rank=dev, device=dev, isTrain=train)
        checkpoint = torch.load(ckpt_path, map_location=f'cuda:{dev}')

        for key in checkpoint.keys():
            if 'model_state' in key and checkpoint[key] is not None:
                consume_prefix_in_state_dict_if_present(checkpoint[key], 'module.')

        pix2pix_model.load_state(checkpoint)
        pix2pix_model.set_eval()
        pix2pix_model.params.checkpoint_path = ckpt_path
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
    data_loaders = [None for _ in range(max(devs) + 1)]
    tp_tms = [None for _ in range(max(devs) + 1)]
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


def setup_data_h5py(params, test_set=False):
    """Load validation or test set.
    """

    if params.orography:
        params.orog = h5py.File(params.orography_path, 'r')['orog'][0:720].astype(np.float32)
    else:
        params.orog = None

    data_name = 'test' if test_set else 'validation'

    # load the test wind data
    if test_set:
        files_paths = glob.glob(params.inf_data_path + "/*.h5")
    else:
        files_paths = glob.glob(params.valid_data_path + "/*.h5")
    files_paths.sort()

    if params.log_to_screen:
        logging.info(f'Loading {data_name} data')
        logging.info('Path to data: {}'.format(files_paths[0]))
    data_full = h5py.File(files_paths[0], 'r')['fields']

    # load the test precip data
    if test_set:
        path = params.precip + '/out_of_sample'
    else:
        path = params.precip + '/test'
    precip_paths = glob.glob(path + "/*.h5")
    precip_paths.sort()

    if params.log_to_screen:
        logging.info(f'Loading {data_name} precip data')
        logging.info('Path to precip data: {}'.format(precip_paths[0]))

    data_tp_full = h5py.File(precip_paths[0], 'r')['tp']

    return data_full, data_tp_full


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
                        if params.input_nc > n_wind:
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


def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return (x + noise)


def autoregressive_inference(params,
                             ic,
                             inf_data_full,
                             inf_data_tp_full,
                             model,
                             model_wind,
                             n_pert=0,
                             n_level=0.0,
                             bins=300, bins_max=11.,
                             viz=False,
                             viz_ens=False):
    ic = int(ic)
    # initialize global variables
    device = model.device # torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    dt = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    n_history = params.n_history
    img_shape_x = params.img_size[0]
    img_shape_y = params.img_size[1]
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    m = torch.as_tensor(np.load(params.precip_time_means)).to(device)

    # initialize memory for image sequences and RMSE/ACC
    rmse = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    binned_log_l1 = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    tqe = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)

    # wind seqs
    if n_history > 0:
        seq_real = torch.zeros((prediction_length+n_history, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
        seq_pred = torch.zeros((prediction_length+n_history, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)

    # precip sequences
    seq_real_tp = torch.zeros((prediction_length, n_out_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    seq_pred_tp = torch.zeros((prediction_length, n_out_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)

    if viz_ens:
        seq_ens_tp = torch.zeros((n_pert, prediction_length+n_history, n_out_channels, img_shape_x, img_shape_y)).to('cpu', dtype=torch.float)

    pred_hists = torch.zeros((prediction_length, bins)).to(device, dtype=torch.float)
    tar_hists = torch.zeros((prediction_length, bins)).to(device, dtype=torch.float)

    # standardize
    inf_data = inf_data_full[ic:(ic+prediction_length*dt+n_history*dt):dt, in_channels, 0:720] # extract test data from first year
    inf_data = torch.cat([
        torch.unsqueeze(
            reshape_fields(inf_data[i], 'inp',
                           params.crop_size_x, params.crop_size_y,
                           rnd_x=0, rnd_y=0, params=params, y_roll=0,
                           train=False, normalize=True, orog=params.orog),
            dim=0
        )
        for i in range(inf_data.shape[0])
    ], dim=0)
    inf_data = inf_data.to(device)

    # log normalize
    len_ic = prediction_length*dt
    inf_data_tp = inf_data_tp_full[ic:(ic+prediction_length*dt):dt, 0:720].reshape(len_ic,n_out_channels,720,img_shape_y) #extract test data from first year
    inf_data_tp = torch.cat([
        torch.unsqueeze(
            reshape_precip(inf_data_tp[i],
                           params.crop_size_x, params.crop_size_y,
                           rnd_x=0, rnd_y=0, params=params, y_roll=0,
                           train=False, normalize=True),
            dim=0
        )
        for i in range(inf_data.shape[0])
    ], dim=0)
    inf_data_tp = inf_data_tp.to(device)

    if 'prev_precip_input' in params:
        prev_precip_input = params.prev_precip_input
    else:
        prev_precip_input = False

    n_wind = params.afno_wind_N_channels

    if params.log_to_screen:
        logging.info('Begin autoregressive+tp inference')

    for pert in range(max(1, n_pert)):
        if n_pert > 0:
            logging.info('Running ensemble {}/{}'.format(pert+1, n_pert))
        else:
            logging.info('Running control')
        with torch.inference_mode():
            for i in range(inf_data.shape[0]):

                if i == 0:  # start of sequence
                    first = inf_data[0:n_history+1]
                    first_tp = inf_data_tp[0:1]
                    future_tp = inf_data_tp[1]

                    if n_history > 0:
                        for h in range(n_history+1):
                            seq_real[h] = first[h*n_in_channels:(h+1)*n_in_channels][0:n_in_channels]  # extract history from 1st
                            seq_pred[h] = seq_real[h]

                    seq_real_tp[0] = first_tp
                    seq_pred_tp[0] = first_tp

                    if n_level > 0. and n_pert != 0:
                        first = gaussian_perturb(first, level=n_level, device=device)  # perturb the ic

                    future_pred = model_wind(first[:, :n_wind])

                    if prev_precip_input:
                        future_pred = torch.cat([future_pred, first_tp], dim=1)

                    if first.shape[1] > n_wind:
                        future_pred = torch.cat([future_pred, first[:, n_wind:]], dim=1)

                    future_pred_tp, _ = model.generate_fake(future_pred, first_tp, validation=True)

                elif i < prediction_length - 1:
                    future_tp = inf_data_tp[i+1]

                    future_pred_ = model_wind(future_pred[:, :n_wind]) # autoregressive step

                    if prev_precip_input:
                        future_pred_ = torch.cat([future_pred_, future_pred_tp], dim=1)

                    if first.shape[1] > n_wind:
                        future_pred = torch.cat([future_pred_, first[:, n_wind:]], dim=1)
                    else:
                        future_pred = future_pred_

                    future_pred_tp, _ = model.generate_fake(future_pred, first_tp, validation=True)  # tp diagnosis

                if i < prediction_length - 1: # not on the last step
                    if viz_ens:
                        seq_ens_tp[pert, n_history+i+1] = torch.squeeze(future_pred_tp, 0).cpu()

                    # add up predictions and average later
                    seq_pred_tp[n_history+i+1] += unlog_tp_torch(torch.squeeze(future_pred_tp, 0))
                    seq_real_tp[n_history+i+1] += unlog_tp_torch(future_tp)

    # Compute metrics
    for i in range(inf_data.shape[0]):

        if i > 0 and n_pert > 0:
            # avg images
            seq_pred_tp[i] /= n_pert
            seq_real_tp[i] /= n_pert

        pred_unlog = torch.unsqueeze(seq_pred_tp[i], 0)
        tar_unlog = torch.unsqueeze(seq_real_tp[i], 0)
        rmse[i] = weighted_rmse_torch_channels(pred_unlog, tar_unlog)
        acc[i] = weighted_acc_torch_channels(pred_unlog-m, tar_unlog-m)
        tqe[i] = top_quantiles_error_torch(pred_unlog, tar_unlog)

        pred = torch.log1p(pred_unlog/params.precip_eps)
        tar = torch.log1p(tar_unlog/params.precip_eps)
        pred_hist, tar_hist = precip_histc2(pred, tar, bins=bins, max=bins_max)
        pred_hists[i] = pred_hist
        tar_hists[i] = tar_hist
        binned_log_l1[i] = binned_precip_log_l1(pred, tar, pred_hist, tar_hist)
        seq_pred_tp[i] = torch.squeeze(pred, 0)
        seq_real_tp[i] = torch.squeeze(tar, 0)

        if params.log_to_screen:
            log_str = f'Timestep {i} of {prediction_length}. TP RMS Error: {rmse[i,0]:.4f}, ACC: {acc[i,0]:.4f}'
            log_str += f', binned log L1: {binned_log_l1[i,0]:.4f} , TQE: {tqe[i,0]:.4f}'
            logging.info(log_str)

    seq_real_tp = seq_real_tp.cpu().numpy()
    seq_pred_tp = seq_pred_tp.cpu().numpy()
    rmse = rmse.cpu().numpy()
    acc = acc.cpu().numpy()
    binned_log_l1 = binned_log_l1.cpu().numpy()
    tqe = tqe.cpu().numpy()
    pred_hists = pred_hists.cpu().numpy()
    tar_hists = tar_hists.cpu().numpy()

    out = {
        'rmse': np.expand_dims(rmse, axis=0),
        'acc': np.expand_dims(acc, axis=0),
        'binned_log_l1': np.expand_dims(binned_log_l1, axis=0),
        'tqe': np.expand_dims(tqe, axis=0),
        'pred_hists': np.expand_dims(pred_hists, axis=0),
        'tar_hists': np.expand_dims(tar_hists, axis=0),
    }

    if viz:
        out['seq_real_tp'] = np.expand_dims(seq_real_tp, axis=0)
        out['seq_pred_tp'] = np.expand_dims(seq_pred_tp, axis=0)

    if viz_ens:
        out['seq_ens_tp'] = np.expand_dims(seq_ens_tp, axis=0)

    return out


def rollout(run,
            ics,
            n_pert=0,
            n_level=0.0,
            device=0,
            use_best_acc=True,
            use_best_binned_log_l1=False,
            test_set=False,
            viz=False,
            viz_ens=False,
            root_path='/pscratch/sd/j/jpduncan/weatherbenching/ERA5_generative',
            base_params=None,
            log_to_screen=False):
    """AR rollout for validation or test data.
    """

    # torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    model = setup_replicas(run, devs=[device],
                           use_best_acc=use_best_acc,
                           use_best_binned_log_l1=use_best_binned_log_l1,
                           root_path=root_path,
                           base_params=base_params)[device]
    params = model.params

    if log_to_screen:
        params.log()

    if log_to_screen:
        logging.info('Loaded trained model checkpoint from {}'.format(params.checkpoint_path))

    assert params.train_on_afno_wind, 'model must be trained on afno wind'

    if log_to_screen:
        logging.info('Loading ANFO wind model from {}'.format(params.afno_model_wind_path))

    afno_wind = setup_afno_wind(params, devs=[device])[device]

    # get data
    data_full, data_tp_full = setup_data_h5py(params, test_set=test_set)

    # initialize dict for image sequences and metrics
    ar_out = {
        'ics': []
    }

    n_ics = len(ics)

    for i, ic in enumerate(ics):
        with torch.inference_mode():
            t1 = time.time()
            logging.info("Initial condition {} of {}".format(i+1, n_ics))
            ar_out_ = autoregressive_inference(params=params,
                                               ic=ic,
                                               inf_data_full=data_full,
                                               inf_data_tp_full=data_tp_full,
                                               model=model,
                                               model_wind=afno_wind,
                                               n_pert=n_pert,
                                               n_level=n_level,
                                               viz=viz,
                                               viz_ens=viz_ens)

            for key, val in ar_out_.items():
                ar_out.setdefault(key, []).append(val)

            ar_out['ics'].append(ic)

            t2 = time.time()-t1
            logging.info("Time for rollout for ic {} = {}".format(i, t2))

    for key, val in ar_out.items():
        if key == 'ics':
            ar_out[key] = np.array(val)
        else:
            ar_out[key] = np.concatenate(val, axis=0)

    return ar_out


class H5DatasetCollectionICs(object):

    def __init__(self, h5_dsets, ics):
        self.ics = ics
        self.dsets = h5_dsets
        self.ics_idx = []
        for i, ic in enumerate(self.ics):
            dset = None
            tot_ics = 0
            for dset in self.dsets:
                if i < tot_ics + dset.shape[0]:
                    break
                else:
                    tot_ics += dset.shape[0]
            assert dset is not None, f'couldn\'t find IC {ic} in dsets!'
            idx_in_dset = i - tot_ics
            self.ics_idx.append((dset, idx_in_dset))
        self.shape = (len(ics), *self.dsets[0].shape[1:])

    def __getitem__(self, key):
        ics = self.ics_idx[key]
        if isinstance(ics, tuple):
            dset, idx_in_dset = ics
            out = ics[0][ics[1]]
        elif isinstance(ics, list):
            out = []
            for dset, idx_in_dset in ics:
                out.append(dset[idx_in_dset:idx_in_dset+1])
            out = np.concatenate(out, axis=0)
        return out


class H5FileCollectionICs(object):

    def __init__(self, h5_files, config_name, inf_name):
        self.config_name = config_name
        self.inf_name = inf_name
        self.files = h5_files
        self.filenames = [file.filename for file in self.files]
        self.filename = self.filenames[0]
        if len(self.filenames) > 1:
            self.filename += f' + {len(self.filenames) - 1} more, see "filenames" attribute'
        dset_dict = {}
        for file in self.files:
            if self.config_name != 'afno':
                assert 'ics' in file.keys(), f'ics not found in h5py.File {file.filename}'
            for key, dset in file.items():
                dset_dict.setdefault(key, [])
                if self.config_name != 'afno' and key not in ['seq_pred_tp', 'seq_real_tp']:
                    dset_dict[key].append(dset[:])
                else:
                    dset_dict[key].append(dset)
        for key, val in dset_dict.items():
            if self.config_name != key not in ['seq_pred_tp', 'seq_real_tp']:
                dset_dict[key] = np.concatenate(val, axis=0)
        if config_name != 'afno' and 'seq_pred_tp' in dset_dict.keys():
            dset_dict['seq_pred_tp'] = H5DatasetCollectionICs(dset_dict['seq_pred_tp'], dset_dict['ics'])
            dset_dict['seq_real_tp'] = H5DatasetCollectionICs(dset_dict['seq_real_tp'], dset_dict['ics'])
        self.dset_dict = dset_dict

    def __getitem__(self, key):
        return self.dset_dict[key]

    def __len__(self):
        return len(self.dset_dict)

    def __contains__(self, key):
        return key in self.dset_dict

    def __iter__(self):
        return iter(self.dset_dict)

    def keys(self):
        return self.dset_dict.keys()

    def values(self):
        return self.dset_dict.values()

    def items(self):
        return self.dset_dict.items()

    def close(self):
        for file in self.files:
            file.close()


def load_inference_results(configs,
                           root_dir: str = '/pscratch/sd/j/jpduncan/weatherbenching/ERA5_generative/inference',
                           viz: bool = False):
    """
    config_names: a list of config names or a dictionary like { 'config': { 'control': 'filename_or_suffix',
                                                                            'ens100': 'filename_or_suffix' } }
    root_dir: path to directory where inference_ensemble.py outputs are saved
    """
    res_dir = 'inference_ensemble'
    if viz:
        res_dir = res_dir + '_viz'

    results = {}

    configs_is_dict = False
    if isinstance(configs, dict):
        config_names = configs.keys()
        configs_is_dict = True
    else:
        config_names = configs

    for config_name in config_names:

        inner_dict = results.setdefault(config_name, {})

        inf_dir = os.path.join(root_dir, config_name, res_dir)

        if configs_is_dict:
            subdirs = configs[config_name].keys()
        else:
            subdirs = next(os.walk(inf_dir))[1]

        for subdir in subdirs:
            inner = inner_dict.setdefault(subdir, [])
            if configs_is_dict:
                if os.path.isfile(configs[config_name][subdir]):
                    fnames = [configs[config_name][subdir]]
                else:
                    assert os.path.isdir(inf_dir), f'{inf_dir} doesn\'t exist'
                    h5_glob = str(os.path.join(inf_dir, subdir, f'*{configs[config_name][subdir]}'))
                    fnames = glob.glob(h5_glob)
                    fnames.sort()
            else:
                assert os.path.isdir(inf_dir), f'{inf_dir} doesn\'t exist'
                h5_glob = str(os.path.join(inf_dir, subdir, '*.h5'))
                fnames = glob.glob(h5_glob)
                fnames.sort()
            for fname in fnames:
                inner.append(h5py.File(fname, 'r'))

            inner_dict[subdir] = H5FileCollectionICs(inner_dict[subdir],
                                                     config_name=config_name,
                                                     inf_name=subdir)

    return results


def concat_inference_results(base_dir, subdir=''):
    h5_glob = str(os.path.join(base_dir, '*.h5'))
    fnames = glob.glob(h5_glob)
    fnames.sort()
    files = [h5py.File(fname, 'r') for fname in fnames]
    dsets = {}
    ics = []
    # TODO: this seems to have a bug
    for f in files:
        ics.extend(f['ics'][:])
        for key in f.keys():
            dsets.setdefault(key, [])
            dsets[key].append(f[key])

    ics = np.array(ics)
    sort_idx = np.argsort(ics)
    tot_ics = len(ics)
    save_dir = os.path.join(base_dir, subdir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    fstart = -fnames[0][-1::-1].find('/')
    h5fname = os.path.join(save_dir, fnames[0][fstart:][3:])
    with h5py.File(h5fname, 'w') as f_:
        for key, dsets_ in dsets.items():
            if key == 'ics':
                dset_shape = (tot_ics,)
            else:
                dset_shape = (tot_ics, *dsets_[0].shape[1:])
            dset = f_.create_dataset(key, shape=dset_shape, dtype=np.float32)
            start = 0
            for dset_ in dsets_:
                dset[start:(start+dset_.shape[0])] = dset_
                start += dset_.shape[0]
            dset[...] = dset[:][sort_idx]

    return h5py.File(h5fname, 'r')


def gauss_blur_rollout(ics,
                       h5py_file,
                       sigmax=10.,
                       sigby=0.5,
                       n_dt=None,
                       device=0,
                       viz=False,
                       log_to_screen=False):
    """Apply gaussian blurring to AR rollout results
    """

    assert 'seq_pred_tp' in h5py_file.keys(), f'h5py_file at {h5py_file.filename} doesn\'t have key "seq_pred_tp"'

    for ic in ics:
        assert ic in h5py_file['ics'][:], f'ic {ic} missing from stored inference viz'

    sigmas = np.arange(0., sigmax, sigby)
    n_sigmas = len(sigmas)
    n_ics = len(ics)

    if log_to_screen:
        logging.info(f"seq_pred_tp.shape: {h5py_file['seq_pred_tp'].shape}")

    if n_dt is None:
        n_dt = h5py_file['seq_pred_tp'].shape[1]
    else:
        n_pred = h5py_file['seq_pred_tp'].shape[1]
        assert n_dt <= n_pred, f'n_dt should be less than number of preds={n_pred}'

    n_dt = int(n_dt)

    # initialize dict for image sequences and metrics
    blur_out = {
        'ics': [],
        'sigmas': sigmas,
        'acc': torch.zeros((n_ics, n_sigmas, n_dt)).to('cpu', dtype=torch.float),
        'rmse': torch.zeros((n_ics, n_sigmas, n_dt)).to('cpu', dtype=torch.float),
        'tqe': torch.zeros((n_ics, n_sigmas, n_dt)).to('cpu', dtype=torch.float),
    }

    tp_tm = torch.as_tensor(np.load('/pscratch/sd/p/pharring/ERA5/precip/total_precipitation/time_means.npy')).to(f'cuda:{device}')
    ics_h5py = h5py_file['ics'][:]
    pred_ics = h5py_file['seq_pred_tp']
    tar_ics = h5py_file['seq_real_tp']

    # get the index for each of the full ics that are in the input ics
    ic_idxs = [i for i, ic in enumerate(ics_h5py) if ic in ics]

    if viz:
        blur_out['seq_blur_tp'] = torch.zeros((n_ics, n_sigmas, n_dt,
                                               tar_ics.shape[-2], tar_ics.shape[-1])).to('cpu', dtype=torch.float)

    if log_to_screen:
        logging.info(f'tp_tm.shape: {tp_tm.shape}')
        logging.info(f'pred_ics.shape: {pred_ics.shape}')
        logging.info(f'tar_ics.shape: {tar_ics.shape}')

    with torch.inference_mode():
        for i, idx in enumerate(ic_idxs):

            logging.info("Initial condition {} of {}".format(i+1, n_ics))
            t1 = time.time()

            blur_out['ics'].append(ics_h5py[idx])
            pred_seq = torch.as_tensor(pred_ics[idx, :n_dt, :]).to(f'cuda:{device}')
            tar_seq = torch.as_tensor(tar_ics[idx, :n_dt, :]).to(f'cuda:{device}')

            if log_to_screen:
                logging.info(f'pred_seq.shape: {pred_seq.shape}')
                logging.info(f'tar_seq.shape: {tar_seq.shape}')

            unlog_tar = unlog_tp_torch(tar_seq)

            for j, sigma in enumerate(sigmas):

                if log_to_screen:
                    logging.info(f"Sigma of gaussian blur: {sigma}")

                if sigma == 0.:
                    blur_seq = pred_seq
                else:
                    blur_seq = gaussian_blur(pred_seq, kernel_size=9, sigma=sigma)

                if log_to_screen:
                    logging.info(f"blur_seq.shape: {blur_seq.shape}")

                unlog_pred = unlog_tp_torch(blur_seq)
                blur_out['acc'][i, j] = torch.squeeze(weighted_acc_torch_channels(unlog_pred - tp_tm, unlog_tar - tp_tm))
                blur_out['rmse'][i, j] = torch.squeeze(weighted_rmse_torch_channels(unlog_pred, unlog_tar))
                blur_out['tqe'][i, j] = torch.squeeze(top_quantiles_error_torch(unlog_pred, unlog_tar))

                if log_to_screen and i == 0 and j == 1:
                    for key in blur_out.items():
                        logging.info(f"blur_out['acc'][0, 1].shape: {blur_out['acc'][i, j].shape}")
                        logging.info(f"blur_out['rmse'][0, 1].shape: {blur_out['rmse'][i, j].shape}")
                        logging.info(f"blur_out['tqe'][0, 1].shape: {blur_out['tqe'][i, j].shape}")

                if viz:
                    blur_out['seq_blur_tp'][i, j] = torch.squeeze(blur_seq)

            logging.info(f"acc by sigmas={sigmas} for ic {blur_out['ics'][-1]}: {blur_out['acc'][i]}")

            t2 = time.time()-t1
            logging.info("Time for gauss blur for ic {} = {}".format(i, t2))

    for key, val in blur_out.items():
        if key not in ['ics', 'sigmas']:
            blur_out[key] = val.numpy()
        elif key == 'ics':
            blur_out[key] = np.array(val)
        if log_to_screen:
            logging.info(f'final {key} shape: {blur_out[key].shape}')

    return blur_out
