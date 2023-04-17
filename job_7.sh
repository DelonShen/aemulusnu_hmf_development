#!/bin/bash
#SBATCH --job-name=compute-ml-fit-7
#SBATCH --output=logs/2023-04-16-compute-ml-mcmc-fit-7.out
#SBATCH --error=logs/2023-04-16-compute-ml-mcmc-fit-7.err
#SBATCH --time=720:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction
python computeML-MCMC-fit.py Box70_1400
python computeML-MCMC-fit.py Box71_1400
python computeML-MCMC-fit.py Box72_1400
python computeML-MCMC-fit.py Box73_1400
python computeML-MCMC-fit.py Box74_1400
python computeML-MCMC-fit.py Box75_1400
python computeML-MCMC-fit.py Box76_1400
python computeML-MCMC-fit.py Box77_1400
python computeML-MCMC-fit.py Box78_1400
python computeML-MCMC-fit.py Box79_1400
