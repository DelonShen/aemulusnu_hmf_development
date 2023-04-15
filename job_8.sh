#!/bin/bash
#SBATCH --job-name=compute-ml-fit-8
#SBATCH --output=logs/2023-04-15-compute-ml-fit-8.out
#SBATCH --error=logs/2023-04-15-compute-ml-fit-8.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate massfunction
python computeMLFit.py Box80_1400
python computeMLFit.py Box81_1400
python computeMLFit.py Box82_1400
python computeMLFit.py Box83_1400
python computeMLFit.py Box84_1400
python computeMLFit.py Box85_1400
python computeMLFit.py Box86_1400
python computeMLFit.py Box87_1400
python computeMLFit.py Box88_1400
python computeMLFit.py Box89_1400
