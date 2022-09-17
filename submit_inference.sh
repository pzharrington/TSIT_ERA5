#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -C gpu
#SBATCH --account=m4134_g
#SBATCH -q regular_ss11
#SBATCH --image=nersc/pytorch:ngc-22.02-v0

ROOT_DIR=$SCRATCH/weatherbenching/ERA5_generative
IMAGE=nersc/pytorch:ngc-22.02-v0
PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-22.02-v0

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(hostname)

args="${@}"

set -x
srun --mpi=pmi2 -u -l shifter \
     --module gpu --env PYTHONUSERBASE=$PYTHONUSERBASE \
     bash -c "
      source export_DDP_vars.sh
      python inference/inference_ensemble.py \
             --root_dir=${ROOT_DIR} ${args}
      "
