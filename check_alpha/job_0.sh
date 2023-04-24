#!/bin/bash
#SBATCH --job-name=prepare-data-0
#SBATCH --output=2023-04-23-prepare-data-0.out
#SBATCH --error=2023-04-23-prepare-data-0.err
#SBATCH --time=60:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python prepare_data.py Box000
