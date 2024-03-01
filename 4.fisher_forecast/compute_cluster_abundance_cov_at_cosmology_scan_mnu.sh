#!/bin/bash



mnus=(0.06)
#for i in $(seq 0.06 0.01 .302); do
#  mnus+=($i)
#done
#
# mnus=(0.062)


for((i=0; i<${#mnus[@]}; i++)); do
        mnu="${mnus[i]}"
        step_size="${step_sizes[j]}" 
        echo $mnu
        # Generate job name with index
        job_name="compute_cov_$mnu"
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
#SBATCH --time=240:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

conda init
conda activate massfunction

python -u compute_cluster_abundance_covariance.py $mnu


EOF
done
