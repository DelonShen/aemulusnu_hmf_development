#!/bin/bash
#SBATCH --job-name=spatial-jackknife-and-NvsM-n50-3
#SBATCH --output=logs/2023-04-19-jack-and-NvsM-n50-3.out
#SBATCH --error=logs/2023-04-19-jack-and-NvsM-n50-3.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python spatial_jackknife_and_NvsM.py Box_n50_30_1400
python spatial_jackknife_and_NvsM.py Box_n50_31_1400
python spatial_jackknife_and_NvsM.py Box_n50_32_1400
python spatial_jackknife_and_NvsM.py Box_n50_33_1400
python spatial_jackknife_and_NvsM.py Box_n50_34_1400
python spatial_jackknife_and_NvsM.py Box_n50_35_1400
python spatial_jackknife_and_NvsM.py Box_n50_36_1400
python spatial_jackknife_and_NvsM.py Box_n50_37_1400
python spatial_jackknife_and_NvsM.py Box_n50_38_1400
python spatial_jackknife_and_NvsM.py Box_n50_39_1400
