#!/bin/bash

#SBATCH --job-name=total_gen
#SBATCH --error=slurms/total_gen.err
#SBATCH --output=slurms/total_gen.out

jid=$(sbatch MC_script.sh) && sbatch --dependency=afterok:${jid##* } compress.sh