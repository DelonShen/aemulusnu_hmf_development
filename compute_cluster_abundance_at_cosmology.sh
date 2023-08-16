#!/bin/bash

param_names=('10^9 As' 'ns' 'H0' 'w0' 'ombh2' 'omch2' 'nu_mass_ev' 'sigma8')

param_names=('ombh2')
step_sizes=() # Initialize the array

for i in $(seq 1.4 0.025 2.4); do
  step_sizes+=($i)
done

for ((i=0; i<${#param_names[@]}; i++)); do
    for ((j=0; j<${#step_sizes[@]}; j++)); do
        param="${param_names[i]}" # Access the element at index i
        step_size="${step_sizes[j]}" # Access the element at index j
        echo $param
        echo -$step_size
        # Generate job name with index
        job_name="computeN_finer_"$param"_$step_size"
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
#SBATCH --time=10:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=32

conda init
conda activate massfunction

python compute_cluster_abundance_at_cosmology.py "$param" -$step_size


EOF
done
done