#!/bin/bash
#SBATCH --job-name=spatial-jackknife-and-NvsM-6
#SBATCH --output=logs/2023-04-18-jack-and-NvsM-6.out
#SBATCH --error=logs/2023-04-18-jack-and-NvsM-6.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python spatial_jackknife_and_NvsM.py Box60_1400
python spatial_jackknife_and_NvsM.py Box61_1400
python spatial_jackknife_and_NvsM.py Box62_1400
python spatial_jackknife_and_NvsM.py Box63_1400
python spatial_jackknife_and_NvsM.py Box64_1400
python spatial_jackknife_and_NvsM.py Box65_1400
python spatial_jackknife_and_NvsM.py Box66_1400
python spatial_jackknife_and_NvsM.py Box67_1400
python spatial_jackknife_and_NvsM.py Box68_1400
python spatial_jackknife_and_NvsM.py Box69_1400
