#!/bin/bash
#SBATCH --job-name=predictCom
# Job name
#SBATCH --mem=10000
# Job memory request
#SBATCH -t 0-03:00
# Time limit hrs:min:sec
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=dcc-tdunn-gpu-01
module load Anaconda3/5.1.0
source activate dannce
com-predict-single-batch "$@"
