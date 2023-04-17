#!/bin/bash
#SBATCH --job-name=compute-ml-fit-Box4_1400
#SBATCH --output=logs/2023-04-16-compute-ml-fit-Box4_1400.out
#SBATCH --error=logs/2023-04-16-compute-ml-fit-Box4_1400.err
#SBATCH --time=60:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction

python computeMLFit.py Box4_1400
