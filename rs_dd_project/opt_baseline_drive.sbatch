#!/bin/bash

#SBATCH -t 6:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --mem=8G
#SBATCH --exclusive  # might be needed to prevent freezing - investigate further
#SBATCH -A carney-sjones-condo
#SBATCH -J hnn_core_opt
#SBATCH -o slurm.out
#SBATCH -e slurm.err

module load python/3.9.0 mpi/openmpi_4.0.7_gcc_10.2_slurm22 gcc/10.2 cuda/11.7.1
source ~/envs/hnn_core_mpi_root/bin/activate
python ~/hnn-core/rs_dd_project/opt_baseline_drive.py
deactivate