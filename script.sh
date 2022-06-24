#!/bin/bash

#SBATCH --job-name=t50-60-00
#SBATCH --error=slurms/t50-60-00-%a.err
#SBATCH --output=slurms/t50-60-00-%a.out
#SBATCH --array=0-49

L=60
echo $L
Jd=0.0
echo $Jd
num_temps=50
echo $num_temps

idx=$SLURM_ARRAY_TASK_ID
echo "I am array job number" $idx
python ds.py $L $Jd $idx $num_temps