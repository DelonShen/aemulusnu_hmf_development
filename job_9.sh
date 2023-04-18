#!/bin/bash
#SBATCH --job-name=spatial-jackknife-and-NvsM-9
#SBATCH --output=logs/2023-04-18-jack-and-NvsM-9.out
#SBATCH --error=logs/2023-04-18-jack-and-NvsM-9.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python spatial_jackknife_and_NvsM.py Box90_1400
python spatial_jackknife_and_NvsM.py Box91_1400
python spatial_jackknife_and_NvsM.py Box92_1400
python spatial_jackknife_and_NvsM.py Box93_1400
python spatial_jackknife_and_NvsM.py Box94_1400
python spatial_jackknife_and_NvsM.py Box95_1400
python spatial_jackknife_and_NvsM.py Box96_1400
python spatial_jackknife_and_NvsM.py Box97_1400
python spatial_jackknife_and_NvsM.py Box98_1400
python spatial_jackknife_and_NvsM.py Box99_1400
