#!/bin/bash
#SBATCH --job-name=compute-ml-fit-n50-3
#SBATCH --output=logs/2023-04-16-compute-ml-mcmc-fit-n50-3.out
#SBATCH --error=logs/2023-04-16-compute-ml-mcmc-fit-n50-3.err
#SBATCH --time=720:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction
python computeML-MCMC-fit.py Box_n50_30_1400
python computeML-MCMC-fit.py Box_n50_31_1400
python computeML-MCMC-fit.py Box_n50_32_1400
python computeML-MCMC-fit.py Box_n50_33_1400
python computeML-MCMC-fit.py Box_n50_34_1400
python computeML-MCMC-fit.py Box_n50_35_1400
python computeML-MCMC-fit.py Box_n50_36_1400
python computeML-MCMC-fit.py Box_n50_37_1400
python computeML-MCMC-fit.py Box_n50_38_1400
python computeML-MCMC-fit.py Box_n50_39_1400
