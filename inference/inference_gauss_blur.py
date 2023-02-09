import os
import sys
import h5py
import time
import numpy as np
import logging
import argparse

import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')

from utils import logging_utils
from utils.inference import gauss_blur_rollout

logging_utils.config_logger()

DECORRELATION_TIME = 8 # 2 days for preicp


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--path_to_viz", type=str, required=True)
    parser.add_argument("--save_dir", type=str, help='Path to store inference outputs', required=True)
    parser.add_argument("--viz", action='store_true',
                        help='visualize ics; can be used with --n_ics; ignored if --viz_ics is used')
    parser.add_argument("--viz_ics", default=[], type=int, nargs='*',
                        help='ics to visualize, e.g., --viz_ics 0 8 16 24')
    parser.add_argument("--n_dt", default=None, type=int,
                        help='number of time steps to predict; if None, uses "prediction_length"')
    parser.add_argument("--n_ics", default=None, type=int, help='Number of ICs to run')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

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

    log_to_screen = (world_rank == 0)

    saved_viz = h5py.File(args.path_to_viz, 'r')
    ics = saved_viz['ics'][:]

    viz = args.viz
    if len(args.viz_ics) > 0:
        assert not args.debug, 'No reason to use --debug and --viz_ics at the same time'
        assert args.n_ics is None, 'No reason to use --n_ics and --viz_ics at the same time'
        ics = args.viz_ics
        viz = True
        if log_to_screen:
            logging.info(f"Running viz for ics {ics}")

    dir_name = 'inference_gauss_blur'

    if viz:
        dir_name = dir_name + '_viz'

    # Set up directory
    save_dir = os.path.join(args.save_dir, args.config, dir_name, args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    if log_to_screen:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(save_dir, f'{dir_name}.log'))
        logging_utils.log_versions()
        logging.info('Loaded saved viz from {}'.format(args.path_to_viz))

    h5name = os.path.join(save_dir, f'{world_rank:02d}_gauss_blur.h5')

    append = os.path.isfile(h5name)
    if append and args.overwrite:
        os.remove(h5name)
        append = False

    n_prev = 0
    if not args.overwrite:
        for i in range(world_size):
            fname = os.path.join(save_dir, f'{i:02d}_gauss_blur.h5')
            if os.path.isfile(fname):
                with h5py.File(fname, 'r') as f:
                    prev_ics = f['ics'][:]
                    if fname == h5name:
                        n_prev = len(prev_ics)
                    ics = [ic for ic in ics if ic not in prev_ics]

    n_ics = args.n_ics

    if args.debug:
        n_ics = world_size

    if n_ics is not None:
        ics = ics[:n_ics]

    tot_ics = len(ics)

    if tot_ics == 0:
        logging.info('All ICs already completed')

    else:

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

        blur_out = gauss_blur_rollout(ics=ics,
                                      h5py_file=h5py.File(args.path_to_viz, 'r'),
                                      n_dt=args.n_dt,
                                      device=local_rank,
                                      viz=viz,
                                      log_to_screen=log_to_screen)

        n_sigmas = len(blur_out['sigmas'])

        #save predictions and loss
        if world_size > 1 and dist.is_initialized():

            if log_to_screen:
                logging.info("Saving files at {}".format(h5name))

            with h5py.File(h5name, 'a') as f:

                start = 0

                for key, val in blur_out.items():
                    if key == 'ics':
                        dset_shape = (n_prev + n_ics,)
                        max_shape = (None,)
                    elif key == 'sigmas':
                        dset_shape = (n_sigmas,)
                    else:
                        dset_shape = (n_prev + n_ics, *val.shape[1:])
                        max_shape = (None,*val.shape[1:])
                    if log_to_screen:
                        logging.info(f'{key} dataset shape: {dset_shape}')
                    if append and key != 'sigmas':
                        dset = f[key]
                        dset.resize(n_prev + n_ics, axis=0)
                        dset[(n_prev+start):(n_prev+start+n_ics)] = val
                    elif key != 'sigmas':
                        dset = f.create_dataset(key, shape=dset_shape, maxshape=max_shape, dtype=np.float32, chunks=True)
                        dset[start:start+n_ics] = val
                    else:
                        if 'sigmas' not in f.keys():
                            dset = f.create_dataset(key, shape=dset_shape, dtype=np.float32)
                            dset[...] = val

        else:
            if log_to_screen:
                logging.info("Saving files at {}".format(h5name))

            with h5py.File(h5name, 'a') as f:
                for key, val in blur_out.items():
                    if key == 'ics':
                        dset_shape = (n_prev + n_ics,)
                        max_shape = (None,)
                    else:
                        dset_shape = (n_prev + n_ics, *val.shape[1:])
                        max_shape = (None, *val.shape[1:])
                    if append:
                        dset = f[key]
                        dset.resize(n_prev + n_ics, axis=0)
                        dset[n_prev:] = val
                    else:
                        dset = f.create_dataset(key, data=val, shape=dset_shape, maxshape=max_shape, dtype=np.float32, chunks=True)
                        dset[...] = val

    sys.exit(0)
