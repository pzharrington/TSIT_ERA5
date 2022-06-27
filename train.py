import os, sys, time
import numpy as np
import argparse
import random
import torch
import torchvision

import torch.distributed as dist
import wandb

import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams

from trainers.pix2pix_trainer import Pix2PixTrainer

if __name__ == '__main__':
    # parsers
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/tsit.yaml', type=str)
    parser.add_argument("--config", default='base', type=str)
    parser.add_argument("--root_dir", default='./', type=str, help='root dir to store results')
    parser.add_argument("--amp", action='store_true')
    parser.add_argument("--sweep_id", default=None, type=str, help='sweep config from ./configs/sweeps.yaml')
    parser.add_argument("--group", default=None, type=str, help='group for wandb init')
    # parser.add_argument("--slurm_id", default=None, type=str, help='slurm job ID')
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
     
    trainer = Pix2PixTrainer(params, args)
    if args.sweep_id and trainer.world_rank==0:
        wandb.agent(args.sweep_id, function=trainer.build_and_launch, count=1, project=trainer.params.project)
    else:
        trainer.build_and_launch()

    if dist.is_initialized():
        dist.barrier()
    logging.info('DONE')

