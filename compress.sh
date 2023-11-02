#!/bin/bash

#SBATCH --job-name=compress
#SBATCH --error=slurms/compress.err
#SBATCH --output=slurms/compress.out

srun python3 -u compress.py 