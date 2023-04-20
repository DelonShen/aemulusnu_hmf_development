#!/bin/bash
#SBATCH --job-name=ml+mcmc-fit-sparse-Box29_1400,
#SBATCH --output=logs/2023-04-20-ml+mcmc-fit-sparse-Box29_1400,.out
#SBATCH --error=logs/2023-04-20-ml+mcmc-fit-sparse-Box29_1400,.err
#SBATCH --time=1440:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction

python computeMLFit.py Box29_1400,
python computeML-MCMC-fit.py Box29_1400,
