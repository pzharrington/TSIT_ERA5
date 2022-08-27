import os
import sys
import glob
import time
import h5py
import numpy as np
import logging
import argparse
import torch

import torch.cuda.amp as amp
import torch.distributed as dist
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')

from utils import logging_utils
from utils.inference import setup_afno_wind, setup_replicas
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, \
    weighted_acc_torch_channels, unlog_tp_torch, top_quantiles_error_torch
from utils.precip_hists import precip_histc2, binned_precip_log_l1
from utils.spectra_metrics import spectra_metrics_rfft, spectra_metrics_fft_input
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from utils.img_utils import reshape_fields, reshape_precip
from models.afnonet import AFNONet, PrecipNet, load_afno

from datetime import datetime

logging_utils.config_logger()

DECORRELATION_TIME = 8 # 2 days for preicp


def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return (x + noise)


def setup_test_data(params):

    if params.train_on_afno_wind:
        afno_wind = setup_afno_wind(params, devs=[0])[0]

    in_channels = params.in_channels
    out_channels = params.out_channels
    wind_n_ch = np.array(params.out_channels) # same as in for the wind model

    if params.orography:
        params.orog = h5py.File(params.orography_path, 'r')['orog'][0:720].astype(np.float32)
    else:
        params.orog = None

    # load the test wind data
    files_paths = glob.glob(params.inf_data_path + "/*.h5")
    files_paths.sort()
    if params.log_to_screen:
        logging.info('Loading test data')
        logging.info('Test data from {}'.format(files_paths[0]))
    test_data_full = h5py.File(files_paths[0], 'r')['fields']

    # load the test precip data
    path = params.precip + '/out_of_sample'
    precip_paths = glob.glob(path + "/*.h5")
    precip_paths.sort()

    if params.log_to_screen:
        logging.info('Loading test precip data')
        logging.info('Test data from {}'.format(precip_paths[0]))

    test_data_tp_full = h5py.File(precip_paths[0], 'r')['tp']

    return test_data_full, test_data_tp_full


def autoregressive_inference(params, ic, test_data_full, test_data_tp_full, model_wind, model,
                             bins=300, bins_max=11.):
    ic = int(ic)
    # initialize global variables
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
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

    n_pert = params.n_pert
    n_level = params.n_level

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

    pred_hists = torch.zeros((prediction_length, bins)).to(device, dtype=torch.float)
    tar_hists = torch.zeros((prediction_length, bins)).to(device, dtype=torch.float)

    if n_pert > 0:
        seq_pert_tp = torch.zeros((n_pert, n_out_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)

    # standardize
    test_data = test_data_full[ic:(ic+prediction_length*dt+n_history*dt):dt, in_channels, 0:720] # extract test data from first year
    test_data = torch.cat([
        torch.unsqueeze(
            reshape_fields(test_data[i], 'inp',
                           params.crop_size_x, params.crop_size_y,
                           rnd_x=0, rnd_y=0, params=params, y_roll=0,
                           train=False, normalize=True, orog=params.orog),
            dim=0
        )
        for i in range(test_data.shape[0])
    ], dim=0)
    test_data = test_data.to(device)

    # log normalize
    len_ic = prediction_length*dt
    test_data_tp = test_data_tp_full[ic:(ic+prediction_length*dt):dt, 0:720].reshape(len_ic,n_out_channels,720,img_shape_y) #extract test data from first year
    test_data_tp = torch.cat([
        torch.unsqueeze(
            reshape_precip(test_data_tp[i], 'tar',
                           params.crop_size_x, params.crop_size_y,
                           rnd_x=0, rnd_y=0, params=params, y_roll=0,
                           train=False, normalize=True),
            dim=0
        )
        for i in range(test_data.shape[0])
    ], dim=0)
    test_data_tp = test_data_tp.to(device)

    n_wind = params.afno_wind_N_channels

    if params.log_to_screen:
        logging.info('Begin autoregressive+tp inference')

    for pert in range(max(1, n_pert)):
        if n_pert > 0:
            logging.info('Running ensemble {}/{}'.format(pert+1, n_pert))
        else:
            logging.info('Running control')
        with torch.inference_mode():
            for i in range(test_data.shape[0]):
                if i == 0:  # start of sequence
                    first = test_data[0:n_history+1]
                    first_tp = test_data_tp[0:1]
                    future = test_data[n_history+1]
                    future_tp = test_data_tp[1]
                    if n_history > 0:
                        for h in range(n_history+1):
                            seq_real[h] = first[h*n_in_channels:(h+1)*n_in_channels][0:n_in_channels]  # extract history from 1st
                            seq_pred[h] = seq_real[h]
                    seq_real_tp[0] = first_tp
                    seq_pred_tp[0] = first_tp
                    if n_level > 0. and n_pert != 0:
                        first = gaussian_perturb(first, level=n_level, device=device)  # perturb the ic
                    future_pred = model_wind(first[:, :n_wind])
                    if params.add_grid or params.orography:
                        future_pred = torch.cat([future_pred, first[:, n_wind:]], dim=1)
                    future_pred_tp, _ = model.generate_fake(future_pred, first_tp)
                else:
                    if i < prediction_length - 1:
                        future = test_data[n_history+i+1]
                        future_tp = test_data_tp[i+1]

                    future_pred_ = model_wind(future_pred[:, :n_wind]) # autoregressive step
                    if params.add_grid or params.orography:
                        future_pred = torch.cat([future_pred_, future_pred[:, n_wind:]], dim=1)
                    else:
                        future_pred = future_pred_
                    future_pred_tp, _ = model.generate_fake(future_pred, first_tp)  # tp diagnosis
                    if i == 1 and n_pert > 0:
                        seq_pert_tp[pert] = future_pred_tp

                if i < prediction_length - 1: # not on the last step
                    # add up predictions and average later
                    seq_pred_tp[n_history+i+1] += torch.squeeze(future_pred_tp, 0)
                    seq_real_tp[n_history+i+1] += future_tp

    # Compute metrics
    for i in range(test_data.shape[0]):

        if i > 0 and n_pert > 0:
            # avg images
            seq_pred_tp[i] /= n_pert
            seq_real_tp[i] /= n_pert

        pred = torch.unsqueeze(seq_pred_tp[i], 0)
        tar = torch.unsqueeze(seq_real_tp[i], 0)
        pred_unlog = unlog_tp_torch(pred)
        tar_unlog = unlog_tp_torch(tar)
        rmse[i] = weighted_rmse_torch_channels(pred_unlog, tar_unlog)
        acc[i] = weighted_acc_torch_channels(pred_unlog-m, tar_unlog-m)
        pred_hist, tar_hist = precip_histc2(pred, tar, bins=bins, max=bins_max)
        pred_hists[i] = pred_hist
        tar_hists[i] = tar_hist
        binned_log_l1[i] = binned_precip_log_l1(pred, tar, pred_hist, tar_hist)
        tqe[i] = top_quantiles_error_torch(pred_unlog, tar_unlog)

        if params.log_to_screen:
            log_str = f'Timestep {i} of {prediction_length}. TP RMS Error: {rmse[i,0]:.4f}, ACC: {acc[i,0]:.4f}'
            log_str += f', binned log L1: {binned_log_l1[i,0]:.4f} , TQE: {tqe[i,0]:.4f}'
            logging.info(log_str)

    seq_real_tp = seq_real_tp.cpu().numpy()
    seq_pred_tp = seq_pred_tp.cpu().numpy()
    if n_pert > 0:
        seq_pert_tp = seq_pert_tp.cpu().numpy()
    rmse = rmse.cpu().numpy()
    acc = acc.cpu().numpy()
    binned_log_l1 = binned_log_l1.cpu().numpy()
    tqe = tqe.cpu().numpy()
    pred_hists = pred_hists.cpu().numpy()
    tar_hists = tar_hists.cpu().numpy()

    out = {
        'seq_real_tp': np.expand_dims(seq_real_tp, axis=0),
        'seq_pred_tp': np.expand_dims(seq_pred_tp, axis=0),
        'rmse': np.expand_dims(rmse, axis=0),
        'acc': np.expand_dims(acc, axis=0),
        'binned_log_l1': np.expand_dims(binned_log_l1, axis=0),
        'tqe': np.expand_dims(tqe, axis=0),
        'pred_hists': np.expand_dims(pred_hists, axis=0),
        'tar_hists': np.expand_dims(tar_hists, axis=0),
    }
    if n_pert > 0:
        out['seq_pert_tp'] = np.expand_dims(seq_pert_tp, axis=0)

    return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/tsit.yaml', type=str)
    parser.add_argument("--config", default='inf_l1_only_afno_wind', type=str)
    parser.add_argument("--n_level", default=0.0, type=float)
    parser.add_argument("--n_pert", default=100, type=int)
    parser.add_argument("--override_dir", default=None, type=str, help='Path to store inference outputs')
    parser.add_argument("--use_best_acc", action='store_true')
    parser.add_argument("--use_best_binned_log_l1", action='store_true')
    parser.add_argument("--n_ics", default=None, type=int, help='Number of ICs to run')
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    world_size = 1
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])

    world_rank = 0
    local_rank = 0

    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend='nccl', init_method='env://')
        args.gpu = local_rank
        world_rank = dist.get_rank()
        world_size = dist.get_world_size()

    model = setup_replicas(params.run, devs=[local_rank],
                           use_best_acc=args.use_best_acc,
                           use_best_binned_log_l1=args.use_best_binned_log_l1,
                           base_params=params)[local_rank]
    device = model.device
    params = model.params

    params['local_rank'] = local_rank
    params['world_rank'] = world_rank
    params['world_size'] = world_size
    params.log_to_screen = params.log_to_screen and world_rank == 0
    if params.log_to_screen:
        logging.info('Loaded trained model checkpoint from {}'.format(params.checkpoint_path))


    assert params.train_on_afno_wind, 'model must be trained on afno wind'

    if params.log_to_screen:
        logging.info('Loading ANFO wind model from {}'.format(params.afno_model_wind_path))

    afno_wind = setup_afno_wind(params, devs=[local_rank])[local_rank]

    # get data
    test_data_full, test_data_tp_full = setup_test_data(params)

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    # Set up directory
    if args.override_dir is not None:
        save_dir = os.path.join(args.override_dir, args.config, 'inference_ensemble', str(args.run_num))
    else:
        save_dir = os.path.join(params.experiment_dir, 'inference_ensemble', str(args.run_num))

    if world_rank == 0:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    if world_rank == 0:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(save_dir, 'inference_ensemble.log'))
        logging_utils.log_versions()
        params.log()

    num_samples = 1460 - params.prediction_length
    stop = num_samples
    ics = np.arange(0, stop, DECORRELATION_TIME)
    n_ics = len(ics)
    logging.info("Inference for {} initial conditions".format(n_ics))

    try:
        ar_inf_filetag = params["inference_file_tag"]
    except:
        ar_inf_filetag = ""

    n_level = args.n_level
    params.n_level = n_level
    params.n_pert = args.n_pert

    if params.n_pert > 0:
        logging.info("Doing level = {}".format(n_level))
        ar_inf_filetag += "_" + str(params.n_level) + "_" + str(params.n_pert) + "ens_tp"
    else:
        logging.info("Doing control")
        ar_inf_filetag += "_control_tp"

    # initialize dict for image sequences and metrics
    ar_out = {
        'ics': []
    }

    if args.debug:
        ics = ics[0:4]
        n_ics = len(ics)
        ar_inf_filetag += '_DEBUG'

    if args.n_ics is not None:
        n_ics = args.n_ics
        ics = ics[0:n_ics]
        ar_inf_filetag += f'_n_ics{n_ics}'

    if args.use_best_acc:
        ar_inf_filetag += '_ckpt_best'

    if args.use_best_binned_log_l1:
        ar_inf_filetag += '_ckpt_best_binned_log_l1'

    # run autoregressive inference for multiple initial conditions
    # parallelize over initial conditions
    if world_size > 1:
        tot_ics = len(ics)
        ics_per_proc = n_ics//world_size
        ics = ics[ics_per_proc*world_rank:ics_per_proc*(world_rank+1)] if world_rank < world_size - 1 else ics[(world_size - 1)*ics_per_proc:]
        n_ics = len(ics)
        logging.info('Rank %d running ics %s'%(world_rank, str(ics)))
        logging.info(f'World info -- world_size: {world_size}, world_rank: {world_rank}, local_rank: {local_rank}')

    for i, ic in enumerate(ics):
        with torch.inference_mode():
            t1 = time.time()
            logging.info("Initial condition {} of {}".format(i+1, n_ics))
            ar_out_ = autoregressive_inference(params, ic,
                                               test_data_full, test_data_tp_full,
                                               afno_wind, model)

            for key, val in ar_out_.items():
                ar_out.setdefault(key, []).append(val)

            ar_out['ics'].append(ic)

            t2 = time.time()-t1
            logging.info("Time for inference for ic {} = {}".format(i, t2))

    for key, val in ar_out.items():
        if key == 'ics':
            ar_out[key] = np.array(val)
        else:
            ar_out[key] = np.concatenate(val, axis=0)

    prediction_length = ar_out['seq_real_tp'][0].shape[0]
    n_out_channels = ar_out['seq_real_tp'][0].shape[1]
    # logging.info(f'acc.shape: {acc.shape}')

    shape_x, shape_y = params.img_size[0], params.img_size[1]

    fields = {
        'ground_truth': ar_out['seq_real_tp'], # ar_out.pop('seq_real_tp'),
        'predicted': ar_out['seq_pred_tp'], # ar_out.pop('seq_pred_tp'),
    }

    #save predictions and loss
    h5name = os.path.join(save_dir, 'ens_autoregressive_predictions'+ ar_inf_filetag +'.h5')
    if dist.is_initialized():

        if params.log_to_screen:
            logging.info("Saving files at {}".format(h5name))

        dist.barrier()
        from mpi4py import MPI
        with h5py.File(h5name, 'w', driver='mpio', comm=MPI.COMM_WORLD) as f:

            start = world_rank*ics_per_proc

            for key, val in ar_out.items():
                if key == 'ics':
                    dset_shape = tot_ics
                else:
                    dset_shape = (tot_ics, *val.shape[1:])
                if params.log_to_screen:
                    logging.info(f'{key} dataset shape: {dset_shape}')
                f.create_dataset(key, shape=dset_shape, dtype=np.float32)

            for key, val in ar_out.items():
                f[key][start:start+n_ics] = val

        dist.barrier()
    else:
        if params.log_to_screen:
            logging.info("Saving files at {}".format(h5name))

        with h5py.File(h5name, 'a') as f:
            for key, val in ar_out.items():
                if key == 'ics':
                    dset_shape = tot_ics
                else:
                    dset_shape = (tot_ics, *val.shape[1:])
                try:
                    f.create_dataset(key, data=val, shape=dset_shape, dtype=np.float32)
                except:
                    del f[key]
                    f.create_dataset(key, data=val, shape=dset_shape, dtype=np.float32)
                    f[key][...] = val
