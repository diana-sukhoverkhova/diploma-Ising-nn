#!/bin/bash

#SBATCH --job-name=test-1
#SBATCH --cpus-per-task=1
#SBATCH --error=slurms/test-1-%a.err
#SBATCH --output=slurms/test-1-%a.out
#SBATCH --array=1-30

idx=$SLURM_ARRAY_TASK_ID
echo "Start testing for epoch = " $idx
num_temps=100
echo "num_temps = " $num_temps
opt='2'
python3 test.py $idx $num_temps $opt