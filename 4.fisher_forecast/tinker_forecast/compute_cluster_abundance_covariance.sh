# Generate job name with index
job_name="tinker_compute_cov_DESy3"
# Define output and error log file paths
output_log="logs/$(date +%Y-%m-%d)-$job_name.out"
error_log="logs/$(date +%Y-%m-%d)-$job_name.err"
echo $job_name
       
# Submit SLURM job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name="$job_name"
#SBATCH --output="$output_log"
#SBATCH --error="$error_log"
#SBATCH --time=1440:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

conda init
conda activate massfunction

python -u compute_cluster_abundance_cov_tinker.py 0.06


EOF
