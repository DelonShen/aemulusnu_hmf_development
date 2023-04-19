#!/bin/bash
#SBATCH --job-name=compute-ml-fit-9
#SBATCH --output=logs/2023-04-19-compute-ml-fit-9.out
#SBATCH --error=logs/2023-04-19-compute-ml-fit-9.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python computeMLFit.py Box90_1400
python computeMLFit.py Box91_1400
python computeMLFit.py Box92_1400
python computeMLFit.py Box93_1400
python computeMLFit.py Box94_1400
python computeMLFit.py Box95_1400
python computeMLFit.py Box96_1400
python computeMLFit.py Box97_1400
python computeMLFit.py Box98_1400
python computeMLFit.py Box99_1400
