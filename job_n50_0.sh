#!/bin/bash
#SBATCH --job-name=compute-ml-mcmc-fit-n50-0
#SBATCH --output=logs/2023-04-19-compute-ml-mcmc-fit-n50-0.out
#SBATCH --error=logs/2023-04-19-compute-ml-mcmc-fit-n50-0.err
#SBATCH --time=720:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction
python computeML-MCMC-fit.py Box_n50_0_1400
