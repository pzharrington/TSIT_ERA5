import os
import sys
import h5py
import numpy as np
import logging
import argparse

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
    parser.add_argument("--viz", action='store_true',
                        help='visualize ics; can be used with --n_ics; ignored if --viz_ics is used')
    parser.add_argument("--viz_ics", default=[], type=int, nargs='*',
                        help='ics to visualize, e.g., --viz_ics 0 8 16 24')
    parser.add_argument("--viz_ens", action='store_true',
                        help='if true, save all ensemble outputs (probably don\'t use without n_dt / n_ics / viz_ics)')
    parser.add_argument("--n_dt", default=None, type=int,
                        help='number of time steps to predict; if None, uses "prediction_length" from config')
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

    viz = args.viz
    viz_ens = args.viz_ens
    dir_name = 'inference_ensemble'
    if len(args.viz_ics) > 0:
        assert not args.debug, 'No reason to use --debug and --viz_ics at the same time'
        assert args.n_ics is None, 'No reason to use --n_ics and --viz_ics at the same time'
        ics = args.viz_ics
        # if len(viz_ics) < world_size:
        #     # just run a single ic per rank
        #     ics = [ics[0] + i*DECORRELATION_TIME for i in range(world_size)]
        viz = True
        if params.log_to_screen:
            logging.info(f"Running viz for ics {ics}")
    else:
        num_samples = 1460 - params.prediction_length
        stop = num_samples
        ics = np.arange(0, stop, DECORRELATION_TIME)

    if viz_ens:
        dir_name = dir_name + '_viz_ens'
    elif viz:
        dir_name = dir_name + '_viz'

    n_pert = args.n_pert

    if args.n_pert == 0:
        run_name = 'control'
    else:
        run_name = f'ens{n_pert}'

    if args.debug:
        run_name = 'debug'

    # Set up directory
    if args.override_dir is not None:
        save_dir = os.path.join(args.override_dir, args.config, dir_name, run_name)
    else:
        assert False, 'Please use --override_dir'
        # save_dir = os.path.join(params.experiment_dir, dir_name, run_name)

    if args.run_num is not None:
        save_dir = os.path.join(save_dir, str(args.run_num))

    os.makedirs(save_dir, exist_ok=True)

    if params.log_to_screen:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(save_dir, f'{dir_name}.log'))
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

    h5name = os.path.join(save_dir, f'{world_rank:02d}_rollout'+ ar_inf_filetag +'.h5')

    append = os.path.isfile(h5name)
    if append and args.overwrite:
        os.remove(h5name)
        append = False

    n_prev = 0
    if not args.overwrite:
        for i in range(world_size):
            fname = os.path.join(save_dir, f'{i:02d}_rollout'+ ar_inf_filetag + '.h5')
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
        ics = ics[0:n_ics]

    tot_ics = len(ics)

    if args.n_dt is not None:
        assert args.n_dt < params.prediction_length, f'use n_dt < params.prediction_length = {params.prediction_length}'
        params.prediction_length = args.n_dt + 1

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

        ar_out = rollout(run=params.run,
                         ics=ics,
                         n_pert=n_pert,
                         n_level=n_level,
                         device=local_rank,
                         use_best_acc=args.use_best_acc,
                         use_best_binned_log_l1=args.use_best_binned_log_l1,
                         test_set=True,
                         viz=viz,
                         viz_ens=viz_ens,
                         root_path=args.root_dir,
                         base_params=params,
                         log_to_screen=params.log_to_screen)

        #save predictions and loss
        if world_size > 1 and dist.is_initialized():

            if params.log_to_screen:
                logging.info("Saving files at {}".format(h5name))

            with h5py.File(h5name, 'a') as f:

                start = 0

                for key, val in ar_out.items():
                    if key == 'ics':
                        dset_shape = (n_prev + n_ics,)
                        max_shape = (None,)
                    else:
                        dset_shape = (n_prev + n_ics, *val.shape[1:])
                        max_shape = (None,*val.shape[1:])
                    if params.log_to_screen:
                        logging.info(f'{key} dataset shape: {dset_shape}')
                    if append:
                        dset = f[key]
                        dset.resize(n_prev + n_ics, axis=0)
                        dset[(n_prev+start):(n_prev+start+n_ics)] = val
                    else:
                        dset = f.create_dataset(key, shape=dset_shape, maxshape=max_shape, dtype=np.float32, chunks=True)
                        dset[start:start+n_ics] = val

        else:
            if params.log_to_screen:
                logging.info("Saving files at {}".format(h5name))

            with h5py.File(h5name, 'a') as f:
                for key, val in ar_out.items():
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
