import os
import sys
import glob
import time
import h5py
import numpy as np
import logging
import argparse
import torch

import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')

from utils import logging_utils
from utils.inference import rollout
from utils.YParams import YParams

logging_utils.config_logger()

DECORRELATION_TIME = 8 # 2 days for preicp


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default=None, type=str)
    parser.add_argument("--yaml_config", default='./config/tsit.yaml', type=str)
    parser.add_argument("--config", default='inf_l1_only_afno_wind', type=str)
    parser.add_argument("--n_level", default=0.0, type=float)
    parser.add_argument("--n_pert", default=0, type=int)
    parser.add_argument("--root_dir", default='/pscratch/sd/j/jpduncan/weatherbenching/ERA5_generative', type=str,
                        help='Path where runs are saved.')
    parser.add_argument("--override_dir", default=None, type=str, help='Path to store inference outputs')
    parser.add_argument("--use_best_acc", action='store_true')
    parser.add_argument("--use_best_binned_log_l1", action='store_true')
    parser.add_argument("--n_ics", default=None, type=int, help='Number of ICs to run')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    world_rank = 0
    local_rank = 0
    world_size = 1

    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])

    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend='nccl', init_method='env://')
        world_rank = dist.get_rank()
        world_size = dist.get_world_size()

    params.log_to_screen = params.log_to_screen and world_rank == 0
    params['local_rank'] = local_rank
    params['world_rank'] = world_rank
    params['world_size'] = world_size

    num_samples = 1460 - params.prediction_length
    stop = num_samples
    ics = np.arange(0, stop, DECORRELATION_TIME)

    n_pert = args.n_pert

    if args.n_pert == 0:
        run_name = 'control'
    else:
        run_name = f'ens{n_pert}'

    if args.debug:
        run_name = 'debug'

    # Set up directory
    if args.override_dir is not None:
        save_dir = os.path.join(args.override_dir, args.config, 'inference_ensemble', run_name)
    else:
        save_dir = os.path.join(params.experiment_dir, 'inference_ensemble', run_name)

    if args.run_num is not None:
        save_dir = os.path.join(save_dir, str(run_num))

    if world_rank == 0:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    if params.log_to_screen:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(save_dir, 'inference_ensemble.log'))
        logging_utils.log_versions()

    try:
        ar_inf_filetag = "_" + str(params["inference_file_tag"])
    except:
        ar_inf_filetag = ""

    if args.use_best_acc:
        ar_inf_filetag += '_ckpt_best'

    if args.use_best_binned_log_l1:
        ar_inf_filetag += '_ckpt_best_binned_log_l1'

    n_level = args.n_level

    if n_pert > 0:
        logging.info("Doing level = {}".format(n_level))
        ar_inf_filetag += "_level" + str(n_level)
    else:
        logging.info("Doing control")

    h5name = os.path.join(save_dir, 'rollout'+ ar_inf_filetag +'.h5')

    append = os.path.isfile(h5name)
    n_prev = 0
    if append:
        if args.overwrite and world_rank == 0:
            os.remove(h5name)
            append = False
        elif not args.overwrite:
            with h5py.File(h5name, 'r') as f:
                prev_ics = f['ics'][:]
                n_prev = len(prev_ics)
                ics = [ic for ic in ics if ic not in prev_ics]

    n_ics = args.n_ics
    if args.debug:
        n_ics = world_size

    if n_ics is not None:
        ics = ics[0:n_ics]

    tot_ics = len(ics)

    if tot_ics > 0:

        # run autoregressive inference for multiple initial conditions
        # parallelize over initial conditions
        ics_per_proc = tot_ics
        if world_size > 1:
            ics_per_proc = tot_ics//world_size
            ics = ics[ics_per_proc*world_rank:ics_per_proc*(world_rank+1)] if world_rank < world_size - 1 else ics[(world_size - 1)*ics_per_proc:]
            logging.info('Rank %d running ics %s'%(world_rank, str(ics)))
            logging.info(f'World info -- world_size: {world_size}, world_rank: {world_rank}, local_rank: {local_rank}')

        n_ics = len(ics)
        logging.info("Inference for {} initial conditions".format(n_ics))

        ar_out = rollout(run=params.run,
                         ics=ics,
                         n_pert=n_pert,
                         n_level=n_level,
                         device=local_rank,
                         use_best_acc=args.use_best_acc,
                         use_best_binned_log_l1=args.use_best_binned_log_l1,
                         test_set=True,
                         root_path=args.root_dir,
                         base_params=params,
                         log_to_screen=params.log_to_screen)

        #save predictions and loss
        if world_size > 1 and dist.is_initialized():

            if params.log_to_screen:
                logging.info("Saving files at {}".format(h5name))

            dist.barrier()
            from mpi4py import MPI
            with h5py.File(h5name, 'a', driver='mpio', comm=MPI.COMM_WORLD) as f:

                start = world_rank*ics_per_proc

                for key, val in ar_out.items():
                    if key == 'ics':
                        dset_shape = (n_prev + tot_ics,)
                        max_shape = (None,)
                    else:
                        dset_shape = (n_prev + tot_ics, *val.shape[1:])
                        max_shape = (None,*val.shape[1:])
                    if params.log_to_screen:
                        logging.info(f'{key} dataset shape: {dset_shape}')
                    if append:
                        dset = f[key]
                        dset.resize(n_prev + tot_ics, axis=0)
                        dset[(n_prev+start):(n_prev+start+n_ics)] = val
                    else:
                        dset = f.create_dataset(key, shape=dset_shape, maxshape=max_shape, dtype=np.float32, chunks=True)
                        dset[start:start+n_ics] = val

            dist.barrier()
        else:
            if params.log_to_screen:
                logging.info("Saving files at {}".format(h5name))

            with h5py.File(h5name, 'a') as f:
                for key, val in ar_out.items():
                    if key == 'ics':
                        dset_shape = (n_prev + tot_ics,)
                        max_shape = (None,)
                    else:
                        dset_shape = (n_prev + tot_ics, *val.shape[1:])
                        max_shape = (None, *val.shape[1:])
                    if append:
                        dset = f[key]
                        dset.resize(n_prev + tot_ics, axis=0)
                        dset[n_prev:] = val
                    else:
                        dset = f.create_dataset(key, data=val, shape=dset_shape, maxshape=max_shape, dtype=np.float32, chunks=True)
                        dset[...] = val
