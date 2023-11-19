#!/bin/bash

#SBATCH --job-name=m-train_31
#SBATCH --time=2-0:0
#SBATCH --error=slurms/m-train_31_%a.err
#SBATCH --output=slurms/m-train_31_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=20,30,60,80,100,120

idx=$SLURM_ARRAY_TASK_ID
echo "I am array job with L =" $idx
epochs=31
start_epoch=1
num_temps=100
python train.py $epochs $idx $num_temps $start_epoch