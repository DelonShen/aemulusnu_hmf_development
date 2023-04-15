#!/bin/bash
#SBATCH --job-name=compute-ml-fit-n50-0
#SBATCH --output=logs/2023-04-15-compute-ml-fit-n50-0.out
#SBATCH --error=logs/2023-04-15-compute-ml-fit-n50-0.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python computeMLFit.py Box_n50_0_1400
python computeMLFit.py Box_n50_1_1400
python computeMLFit.py Box_n50_2_1400
python computeMLFit.py Box_n50_3_1400
python computeMLFit.py Box_n50_4_1400
python computeMLFit.py Box_n50_5_1400
python computeMLFit.py Box_n50_6_1400
python computeMLFit.py Box_n50_7_1400
python computeMLFit.py Box_n50_8_1400
python computeMLFit.py Box_n50_9_1400
