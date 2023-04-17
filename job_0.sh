#!/bin/bash
#SBATCH --job-name=compute-ml-fit-0
#SBATCH --output=logs/2023-04-16-compute-ml-mcmc-fit-0.out
#SBATCH --error=logs/2023-04-16-compute-ml-mcmc-fit-0.err
#SBATCH --time=720:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction
python computeML-MCMC-fit.py Box0_1400
python computeML-MCMC-fit.py Box1_1400
python computeML-MCMC-fit.py Box2_1400
python computeML-MCMC-fit.py Box3_1400
python computeML-MCMC-fit.py Box4_1400
python computeML-MCMC-fit.py Box5_1400
python computeML-MCMC-fit.py Box6_1400
python computeML-MCMC-fit.py Box7_1400
python computeML-MCMC-fit.py Box8_1400
python computeML-MCMC-fit.py Box9_1400
