#!/bin/bash

#SBATCH --job-name=MC_script
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --error=slurms/MC_script_%a.err
#SBATCH --output=slurms/MC_script_%a.out
#SBATCH --array=0-99

idx=$SLURM_ARRAY_TASK_ID
srun python3 -u MC_script.py $idx
