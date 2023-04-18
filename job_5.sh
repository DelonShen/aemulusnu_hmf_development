#!/bin/bash
#SBATCH --job-name=spatial-jackknife-and-NvsM-5
#SBATCH --output=logs/2023-04-18-jack-and-NvsM-5.out
#SBATCH --error=logs/2023-04-18-jack-and-NvsM-5.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python spatial_jackknife_and_NvsM.py Box50_1400
python spatial_jackknife_and_NvsM.py Box51_1400
python spatial_jackknife_and_NvsM.py Box52_1400
python spatial_jackknife_and_NvsM.py Box53_1400
python spatial_jackknife_and_NvsM.py Box54_1400
python spatial_jackknife_and_NvsM.py Box55_1400
python spatial_jackknife_and_NvsM.py Box56_1400
python spatial_jackknife_and_NvsM.py Box57_1400
python spatial_jackknife_and_NvsM.py Box58_1400
python spatial_jackknife_and_NvsM.py Box59_1400
