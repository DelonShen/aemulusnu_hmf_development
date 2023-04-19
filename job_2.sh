#!/bin/bash
#SBATCH --job-name=compute-ml-fit-2
#SBATCH --output=logs/2023-04-19-compute-ml-fit-2.out
#SBATCH --error=logs/2023-04-19-compute-ml-fit-2.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python computeMLFit.py Box20_1400
python computeMLFit.py Box21_1400
python computeMLFit.py Box22_1400
python computeMLFit.py Box23_1400
python computeMLFit.py Box24_1400
python computeMLFit.py Box25_1400
python computeMLFit.py Box26_1400
python computeMLFit.py Box27_1400
python computeMLFit.py Box28_1400
python computeMLFit.py Box29_1400
