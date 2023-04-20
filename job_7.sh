#!/bin/bash
#SBATCH --job-name=spatial-jackknife-and-NvsM-7
#SBATCH --output=logs/2023-04-19-jack-and-NvsM-7.out
#SBATCH --error=logs/2023-04-19-jack-and-NvsM-7.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python spatial_jackknife_and_NvsM.py Box70_1400
python spatial_jackknife_and_NvsM.py Box71_1400
python spatial_jackknife_and_NvsM.py Box72_1400
python spatial_jackknife_and_NvsM.py Box73_1400
python spatial_jackknife_and_NvsM.py Box74_1400
python spatial_jackknife_and_NvsM.py Box75_1400
python spatial_jackknife_and_NvsM.py Box76_1400
python spatial_jackknife_and_NvsM.py Box77_1400
python spatial_jackknife_and_NvsM.py Box78_1400
python spatial_jackknife_and_NvsM.py Box79_1400
