#!/bin/bash
#SBATCH --job-name=compute-ml-fit-4
#SBATCH --output=logs/2023-04-16-compute-ml-mcmc-fit-4.out
#SBATCH --error=logs/2023-04-16-compute-ml-mcmc-fit-4.err
#SBATCH --time=720:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction
python computeML-MCMC-fit.py Box40_1400
python computeML-MCMC-fit.py Box41_1400
python computeML-MCMC-fit.py Box42_1400
python computeML-MCMC-fit.py Box43_1400
python computeML-MCMC-fit.py Box44_1400
python computeML-MCMC-fit.py Box45_1400
python computeML-MCMC-fit.py Box46_1400
python computeML-MCMC-fit.py Box47_1400
python computeML-MCMC-fit.py Box48_1400
python computeML-MCMC-fit.py Box49_1400
