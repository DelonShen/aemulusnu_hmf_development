#!/bin/bash
#SBATCH --job-name=compute-ml+mcmc-fit-sparse-Box_n50_49_1400
#SBATCH --output=logs/2023-04-20-compute-ml+mcmc-fit-sparse-Box_n50_49_1400.out
#SBATCH --error=logs/2023-04-20-compute-ml+mcmc-fit-sparse-Box_n50_49_1400.err
#SBATCH --time=1440:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction

python computeMLFit.py Box_n50_49_1400
python computeML-MCMC-fit.py Box_n50_49_1400
