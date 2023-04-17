#!/bin/bash
#SBATCH --job-name=compute-ml-fit-6
#SBATCH --output=logs/2023-04-16-compute-ml-mcmc-fit-6.out
#SBATCH --error=logs/2023-04-16-compute-ml-mcmc-fit-6.err
#SBATCH --time=720:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction
python computeML-MCMC-fit.py Box60_1400
python computeML-MCMC-fit.py Box61_1400
python computeML-MCMC-fit.py Box62_1400
python computeML-MCMC-fit.py Box63_1400
python computeML-MCMC-fit.py Box64_1400
python computeML-MCMC-fit.py Box65_1400
python computeML-MCMC-fit.py Box66_1400
python computeML-MCMC-fit.py Box67_1400
python computeML-MCMC-fit.py Box68_1400
python computeML-MCMC-fit.py Box69_1400
