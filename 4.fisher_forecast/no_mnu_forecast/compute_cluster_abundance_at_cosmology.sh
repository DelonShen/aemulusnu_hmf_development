#!/bin/bash

#param_names=('10^9 As' 'ns' 'H0' 'w0' 'ombh2' 'omch2')
#
#param_names=('w0' 'ombh2' 'omch2')
#
##param_names=('ns')
#step_sizes=() # Initialize the array
#
#for i in $(seq 1.6 0.1 2.8); do
#  step_sizes+=($i)
#done
#
#
#
##for ((i=0; i<${#param_names[@]}; i++)); do
##    for ((j=0; j<${#step_sizes[@]}; j++)); do
#
#        param="${param_names[i]}" # Access the element at index i
#        step_size="${step_sizes[j]}" # Access the element at index j
#        echo $param
#        echo -$step_size
#        # Generate job name with index
#        job_name="computeN_"$param"_$step_size"_"no_mnu"
#        # Define output and error log file paths
#        output_log="logs/$(date +%Y-%m-%d)-$job_name.out"
#        error_log="logs/$(date +%Y-%m-%d)-$job_name.err"
#        echo $job_name
#       
#        # Submit SLURM job
#    sbatch <<EOF
##!/bin/bash
##SBATCH --job-name="$job_name"
##SBATCH --output="$output_log"
##SBATCH --error="$error_log"
##SBATCH --time=20:00
##SBATCH -p kipac
##SBATCH --nodes=1
##SBATCH --mem=4096
##SBATCH --cpus-per-task=16
#
#conda init
#conda activate massfunction
#
#python ../compute_cluster_abundance_at_cosmology.py "$param" -$step_size 1 1 1
#
#
#EOF
#done
#done
#

param='omch2'
step_size="2.6"
echo $param
echo -$step_size
# Generate job name with index
job_name="computeN_"$param"_$step_size"_"no_mnu"
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
#SBATCH --time=20:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=4096
#SBATCH --cpus-per-task=16

conda init
conda activate massfunction

python ../compute_cluster_abundance_at_cosmology.py "$param" -$step_size 1 1 1

EOF
