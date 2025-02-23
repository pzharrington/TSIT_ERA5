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
#SBATCH --output=joblogs/slurm-%j.out

# ROOT_DIR=$SCRATCH/weatherbenching/ERA5_generative
ROOT_DIR=/global/cfs/cdirs/dasrepo/jpduncan/weatherbenching/ERA5_generative
PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-22.02-v0

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(hostname)

args="${@}"

set -x
srun -u --mpi=pmi2 \
     shifter --module gpu --env PYTHONUSERBASE=$PYTHONUSERBASE \
     bash -c "
       source export_DDP_vars.sh
       python train.py --root_dir=${ROOT_DIR} ${args}
     "

#shifter --module gpu \
#    bash -c "
#    python train.py --root_dir=${ROOT_DIR} ${args}
#    "
