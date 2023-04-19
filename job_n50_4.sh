#!/bin/bash
#SBATCH --job-name=compute-ml-fit-n50-4
#SBATCH --output=logs/2023-04-19-compute-ml-fit-n50-4.out
#SBATCH --error=logs/2023-04-19-compute-ml-fit-n50-4.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python computeMLFit.py Box_n50_40_1400
python computeMLFit.py Box_n50_41_1400
python computeMLFit.py Box_n50_42_1400
python computeMLFit.py Box_n50_43_1400
python computeMLFit.py Box_n50_44_1400
python computeMLFit.py Box_n50_45_1400
python computeMLFit.py Box_n50_46_1400
python computeMLFit.py Box_n50_47_1400
python computeMLFit.py Box_n50_48_1400
python computeMLFit.py Box_n50_49_1400
