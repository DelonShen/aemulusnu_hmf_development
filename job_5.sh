#!/bin/bash
#SBATCH --job-name=compute-ml-fit-5
#SBATCH --output=logs/2023-04-16-compute-ml-mcmc-fit-5.out
#SBATCH --error=logs/2023-04-16-compute-ml-mcmc-fit-5.err
#SBATCH --time=720:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction
python computeML-MCMC-fit.py Box50_1400
python computeML-MCMC-fit.py Box51_1400
python computeML-MCMC-fit.py Box52_1400
python computeML-MCMC-fit.py Box53_1400
python computeML-MCMC-fit.py Box54_1400
python computeML-MCMC-fit.py Box55_1400
python computeML-MCMC-fit.py Box56_1400
python computeML-MCMC-fit.py Box57_1400
python computeML-MCMC-fit.py Box58_1400
python computeML-MCMC-fit.py Box59_1400
