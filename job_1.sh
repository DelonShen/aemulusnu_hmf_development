#!/bin/bash
#SBATCH --job-name=spatial-jackknife-and-NvsM-n50-1
#SBATCH --output=logs/2023-04-18-jack-and-NvsM-n50-1.out
#SBATCH --error=logs/2023-04-18-jack-and-NvsM-n50-1.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python spatial_jackknife_and_NvsM.py Box_n50_10_1400
python spatial_jackknife_and_NvsM.py Box_n50_11_1400
python spatial_jackknife_and_NvsM.py Box_n50_12_1400
python spatial_jackknife_and_NvsM.py Box_n50_13_1400
python spatial_jackknife_and_NvsM.py Box_n50_14_1400
python spatial_jackknife_and_NvsM.py Box_n50_15_1400
python spatial_jackknife_and_NvsM.py Box_n50_16_1400
python spatial_jackknife_and_NvsM.py Box_n50_17_1400
python spatial_jackknife_and_NvsM.py Box_n50_18_1400
python spatial_jackknife_and_NvsM.py Box_n50_19_1400
