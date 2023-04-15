#!/bin/bash
#SBATCH --job-name=compute-ml-fit-7
#SBATCH --output=logs/2023-04-15-compute-ml-fit-7.out
#SBATCH --error=logs/2023-04-15-compute-ml-fit-7.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python computeMLFit.py Box70_1400
python computeMLFit.py Box71_1400
python computeMLFit.py Box72_1400
python computeMLFit.py Box73_1400
python computeMLFit.py Box74_1400
python computeMLFit.py Box75_1400
python computeMLFit.py Box76_1400
python computeMLFit.py Box77_1400
python computeMLFit.py Box78_1400
python computeMLFit.py Box79_1400
