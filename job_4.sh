#!/bin/bash
#SBATCH --job-name=compute-ml-fit-4
#SBATCH --output=logs/2023-04-15-compute-ml-fit-4.out
#SBATCH --error=logs/2023-04-15-compute-ml-fit-4.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python computeMLFit.py Box40_1400
python computeMLFit.py Box41_1400
python computeMLFit.py Box42_1400
python computeMLFit.py Box43_1400
python computeMLFit.py Box44_1400
python computeMLFit.py Box45_1400
python computeMLFit.py Box46_1400
python computeMLFit.py Box47_1400
python computeMLFit.py Box48_1400
python computeMLFit.py Box49_1400
