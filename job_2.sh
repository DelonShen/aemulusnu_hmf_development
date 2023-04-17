#!/bin/bash
#SBATCH --job-name=compute-ml-fit-2
#SBATCH --output=logs/2023-04-16-compute-ml-mcmc-fit-2.out
#SBATCH --error=logs/2023-04-16-compute-ml-mcmc-fit-2.err
#SBATCH --time=720:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction
python computeML-MCMC-fit.py Box20_1400
python computeML-MCMC-fit.py Box21_1400
python computeML-MCMC-fit.py Box22_1400
python computeML-MCMC-fit.py Box23_1400
python computeML-MCMC-fit.py Box24_1400
python computeML-MCMC-fit.py Box25_1400
python computeML-MCMC-fit.py Box26_1400
python computeML-MCMC-fit.py Box27_1400
python computeML-MCMC-fit.py Box28_1400
python computeML-MCMC-fit.py Box29_1400
