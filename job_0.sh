#!/bin/bash
#SBATCH --job-name=compute-ml-fit-0
#SBATCH --output=logs/2023-04-19-compute-ml-fit-0.out
#SBATCH --error=logs/2023-04-19-compute-ml-fit-0.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python computeMLFit.py Box0_1400
python computeMLFit.py Box1_1400
python computeMLFit.py Box2_1400
python computeMLFit.py Box3_1400
python computeMLFit.py Box4_1400
python computeMLFit.py Box5_1400
python computeMLFit.py Box6_1400
python computeMLFit.py Box7_1400
python computeMLFit.py Box8_1400
python computeMLFit.py Box9_1400
