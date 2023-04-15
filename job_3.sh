#!/bin/bash
#SBATCH --job-name=compute-ml-fit-3
#SBATCH --output=logs/2023-04-15-compute-ml-fit-3.out
#SBATCH --error=logs/2023-04-15-compute-ml-fit-3.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python computeMLFit.py Box30_1400
python computeMLFit.py Box31_1400
python computeMLFit.py Box32_1400
python computeMLFit.py Box33_1400
python computeMLFit.py Box34_1400
python computeMLFit.py Box35_1400
python computeMLFit.py Box36_1400
python computeMLFit.py Box37_1400
python computeMLFit.py Box38_1400
python computeMLFit.py Box39_1400
