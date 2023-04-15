#!/bin/bash
#SBATCH --job-name=compute-ml-fit-n50-2
#SBATCH --output=logs/2023-04-15-compute-ml-fit-n50-2.out
#SBATCH --error=logs/2023-04-15-compute-ml-fit-n50-2.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python computeMLFit.py Box_n50_20_1400
python computeMLFit.py Box_n50_21_1400
python computeMLFit.py Box_n50_22_1400
python computeMLFit.py Box_n50_23_1400
python computeMLFit.py Box_n50_24_1400
python computeMLFit.py Box_n50_25_1400
python computeMLFit.py Box_n50_26_1400
python computeMLFit.py Box_n50_27_1400
python computeMLFit.py Box_n50_28_1400
python computeMLFit.py Box_n50_29_1400
