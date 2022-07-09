#!/bin/bash -l

ROOT_DIR=$SCRATCH/tsitprecip/experiments
IMAGE=nersc/pytorch:ngc-22.02-v0
PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-22.02-v0

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(hostname)

args="${@}"

set -x
srun -u --mpi=pmi2 \
     shifter --module gpu --image=$IMAGE --env PYTHONUSERBASE=$PYTHONUSERBASE \
     bash -c "
       source export_DDP_vars.sh
       python train.py --root_dir=${ROOT_DIR} ${args}
     "

#shifter --module gpu \
    #    bash -c "
#    python train.py --root_dir=${ROOT_DIR} ${args}
#    "
