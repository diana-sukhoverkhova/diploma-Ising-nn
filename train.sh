#!/bin/bash

#SBATCH --job-name=train_100
#SBATCH --time=2-0:0
#SBATCH --error=slurms/train_100_%a.err
#SBATCH --output=slurms/train_100_%a.out
#SBATCH --array=10,20,30 

idx=$SLURM_ARRAY_TASK_ID
echo "I am array job with L =" $idx
epochs=10
num_temps=100
python train.py $epochs $idx $num_temps