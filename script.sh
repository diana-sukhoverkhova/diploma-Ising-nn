#!/bin/bash

#SBATCH --job-name=t100-20-01
#SBATCH --error=slurms/t100-20-01-%a.err
#SBATCH --output=slurms/t100-20-01-%a.out
#SBATCH --array=0-99

L=20
echo $L
Jd=-0.1
echo $Jd
num_temps=100
echo $num_temps

idx=$SLURM_ARRAY_TASK_ID
echo "I am array job number" $idx
python ds.py $L $Jd $idx $num_temps