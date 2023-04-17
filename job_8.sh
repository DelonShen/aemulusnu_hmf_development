#!/bin/bash
#SBATCH --job-name=compute-ml-fit-8
#SBATCH --output=logs/2023-04-16-compute-ml-mcmc-fit-8.out
#SBATCH --error=logs/2023-04-16-compute-ml-mcmc-fit-8.err
#SBATCH --time=720:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction
python computeML-MCMC-fit.py Box80_1400
python computeML-MCMC-fit.py Box81_1400
python computeML-MCMC-fit.py Box82_1400
python computeML-MCMC-fit.py Box83_1400
python computeML-MCMC-fit.py Box84_1400
python computeML-MCMC-fit.py Box85_1400
python computeML-MCMC-fit.py Box86_1400
python computeML-MCMC-fit.py Box87_1400
python computeML-MCMC-fit.py Box88_1400
python computeML-MCMC-fit.py Box89_1400
