#!/bin/bash
#SBATCH --job-name=compute-ml-fit-1
#SBATCH --output=logs/2023-04-16-compute-ml-mcmc-fit-1.out
#SBATCH --error=logs/2023-04-16-compute-ml-mcmc-fit-1.err
#SBATCH --time=720:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction
python computeML-MCMC-fit.py Box10_1400
python computeML-MCMC-fit.py Box11_1400
python computeML-MCMC-fit.py Box12_1400
python computeML-MCMC-fit.py Box13_1400
python computeML-MCMC-fit.py Box14_1400
python computeML-MCMC-fit.py Box15_1400
python computeML-MCMC-fit.py Box16_1400
python computeML-MCMC-fit.py Box17_1400
python computeML-MCMC-fit.py Box18_1400
python computeML-MCMC-fit.py Box19_1400
