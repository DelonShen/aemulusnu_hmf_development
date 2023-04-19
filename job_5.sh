#!/bin/bash
#SBATCH --job-name=compute-ml-fit-5
#SBATCH --output=logs/2023-04-19-compute-ml-fit-5.out
#SBATCH --error=logs/2023-04-19-compute-ml-fit-5.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python computeMLFit.py Box50_1400
python computeMLFit.py Box51_1400
python computeMLFit.py Box52_1400
python computeMLFit.py Box53_1400
python computeMLFit.py Box54_1400
python computeMLFit.py Box55_1400
python computeMLFit.py Box56_1400
python computeMLFit.py Box57_1400
python computeMLFit.py Box58_1400
python computeMLFit.py Box59_1400
