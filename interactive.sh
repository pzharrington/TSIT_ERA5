#!/bin/bash -l

ROOT_DIR=$SCRATCH/tsitprecip/experiments
export HDF5_USE_FILE_LOCKING=FALSE
args="${@}"

export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(hostname)

set -x
srun -u --mpi=pmi2 shifter --module gpu \
     bash -c "
    source export_DDP_vars.sh
    python train.py --root_dir=${ROOT_DIR} ${args}
    "

#shifter --module gpu \
    #    bash -c "
#    python train.py --root_dir=${ROOT_DIR} ${args}
#    "
