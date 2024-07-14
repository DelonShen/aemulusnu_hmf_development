#!/bin/bash

# Generate job name with index
job_name="desi-y1-vary-mnu"
# Define output and error log file paths
output_log="$(date +%Y-%m-%d)-$job_name.out"
error_log="$(date +%Y-%m-%d)-$job_name.err"

# Submit SLURM job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=$output_log
#SBATCH --error=$error_log
#SBATCH --time=168:00:00
#SBATCH -p kipac
#SBATCH -n 8
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

export OMP_NUM_THREADS=4
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun -n 8 -c 4 cobaya-run -r DESI-Y1.yaml

EOF
