#!/bin/bash
#SBATCH --job-name=compute-ml-fit-3
#SBATCH --output=logs/2023-04-16-compute-ml-mcmc-fit-3.out
#SBATCH --error=logs/2023-04-16-compute-ml-mcmc-fit-3.err
#SBATCH --time=720:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction
python computeML-MCMC-fit.py Box30_1400
python computeML-MCMC-fit.py Box31_1400
python computeML-MCMC-fit.py Box32_1400
python computeML-MCMC-fit.py Box33_1400
python computeML-MCMC-fit.py Box34_1400
python computeML-MCMC-fit.py Box35_1400
python computeML-MCMC-fit.py Box36_1400
python computeML-MCMC-fit.py Box37_1400
python computeML-MCMC-fit.py Box38_1400
python computeML-MCMC-fit.py Box39_1400
