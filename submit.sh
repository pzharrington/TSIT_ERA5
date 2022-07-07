#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH -C gpu
#SBATCH --account=m4134_g
#SBATCH -q regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --image=nersc/pytorch:ngc-22.02-v0

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
