#!/bin/bash
#SBATCH --job-name=compute-ml-mcmc-fit-Box39_1400
#SBATCH --output=logs/2023-04-16-compute-ml-mcmc-fit-Box39_1400.out
#SBATCH --error=logs/2023-04-16-compute-ml-mcmc-fit-Box39_1400.err
#SBATCH --time=120:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction

python computeML-MCMC-fit.py Box39_1400
