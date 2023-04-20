#!/bin/bash
#SBATCH --job-name=spatial-jackknife-and-NvsM-8
#SBATCH --output=logs/2023-04-19-jack-and-NvsM-8.out
#SBATCH --error=logs/2023-04-19-jack-and-NvsM-8.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python spatial_jackknife_and_NvsM.py Box80_1400
python spatial_jackknife_and_NvsM.py Box81_1400
python spatial_jackknife_and_NvsM.py Box82_1400
python spatial_jackknife_and_NvsM.py Box83_1400
python spatial_jackknife_and_NvsM.py Box84_1400
python spatial_jackknife_and_NvsM.py Box85_1400
python spatial_jackknife_and_NvsM.py Box86_1400
python spatial_jackknife_and_NvsM.py Box87_1400
python spatial_jackknife_and_NvsM.py Box88_1400
python spatial_jackknife_and_NvsM.py Box89_1400
