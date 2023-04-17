#!/bin/bash
#SBATCH --job-name=compute-ml-fit-9
#SBATCH --output=logs/2023-04-16-compute-ml-mcmc-fit-9.out
#SBATCH --error=logs/2023-04-16-compute-ml-mcmc-fit-9.err
#SBATCH --time=720:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction
python computeML-MCMC-fit.py Box90_1400
python computeML-MCMC-fit.py Box91_1400
python computeML-MCMC-fit.py Box92_1400
python computeML-MCMC-fit.py Box93_1400
python computeML-MCMC-fit.py Box94_1400
python computeML-MCMC-fit.py Box95_1400
python computeML-MCMC-fit.py Box96_1400
python computeML-MCMC-fit.py Box97_1400
python computeML-MCMC-fit.py Box98_1400
python computeML-MCMC-fit.py Box99_1400
