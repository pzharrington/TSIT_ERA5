#!/bin/bash

IMAGE=nersc/pytorch:ngc-22.02-v0
PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-22.02-v0

NGPU=$SLURM_GPUS_ON_NODE

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(hostname)

args="${@}"

set -x
srun --mpi=pmi2 -u -l \
     shifter --module gpu --image=$IMAGE --env PYTHONUSERBASE=$PYTHONUSERBASE \
     bash -c "
       source export_DDP_vars.sh
       python inference/inference_gauss_blur.py ${args}
     "
