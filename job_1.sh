#!/bin/bash
#SBATCH --job-name=compute-ml-fit-1
#SBATCH --output=logs/2023-04-15-compute-ml-fit-1.out
#SBATCH --error=logs/2023-04-15-compute-ml-fit-1.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python computeMLFit.py Box10_1400
python computeMLFit.py Box11_1400
python computeMLFit.py Box12_1400
python computeMLFit.py Box13_1400
python computeMLFit.py Box14_1400
python computeMLFit.py Box15_1400
python computeMLFit.py Box16_1400
python computeMLFit.py Box17_1400
python computeMLFit.py Box18_1400
python computeMLFit.py Box19_1400
