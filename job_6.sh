#!/bin/bash
#SBATCH --job-name=compute-ml-fit-6
#SBATCH --output=logs/2023-04-19-compute-ml-fit-6.out
#SBATCH --error=logs/2023-04-19-compute-ml-fit-6.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python computeMLFit.py Box60_1400
python computeMLFit.py Box61_1400
python computeMLFit.py Box62_1400
python computeMLFit.py Box63_1400
python computeMLFit.py Box64_1400
python computeMLFit.py Box65_1400
python computeMLFit.py Box66_1400
python computeMLFit.py Box67_1400
python computeMLFit.py Box68_1400
python computeMLFit.py Box69_1400
