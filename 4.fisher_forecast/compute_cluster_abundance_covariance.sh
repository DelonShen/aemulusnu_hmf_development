
#!/bin/bash

# Initialize environment for SLURM
for ((i = 0 ; i <= 0 ; i++)); do
    j=$i
        for ((a = 0 ; a <= 0 ; a++)); do
            for ((b = 0 ; b <= 0 ; b++)); do
                
                job_name="compute_clustercount_cov_${i}_${j}_${a}_${b}"
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
#SBATCH --time=30:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=3G
#SBATCH --cpus-per-task=16

# Run Python script
python -u compute_cluster_abundance_covariance.py $i $j $a $b
EOF
            done
        done
done
