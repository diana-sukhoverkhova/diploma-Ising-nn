#!/bin/bash

#SBATCH --job-name=train_10_epochs-80
#SBATCH --error=slurms/train_10_epochs-80.err
#SBATCH --output=slurms/train_10_epochs-80.out
#SBATCH --time=7-0:0

epochs=10
num_temps=100
L=80
python train.py $epochs $L $num_temps 