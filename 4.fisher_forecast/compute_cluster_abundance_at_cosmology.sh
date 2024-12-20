#!/bin/bash


step_sizes=() # Initialize the array


# param_names=('10^9 As' 'nu_mass_ev')
# for i in $(seq 2 0.02 2.3); do
#   step_sizes+=($i)
# done



# param_names=( 'nu_mass_ev')
# for i in $(seq 2.3 0.02 3); do
#   step_sizes+=($i)
# done


#####BLOCK A = PLANCK
#####BLOCK B = DES Y3


#####BLOCK 1
# param_names=('10^9 As' 'H0' 'ombh2' 'omch2')
# for i in $(seq 1 0.05 2.5); do
#   step_sizes+=($i)
# done


# param_names=('ns')

# for i in $(seq 1 0.2 5); do
#  step_sizes+=($i)
# done



#####BLOCK 2
param_names=( 'nu_mass_ev')
for i in $(seq 1.7 0.005 2.3); do
  step_sizes+=($i)
done



for ((i=0; i<${#param_names[@]}; i++)); do
    for ((j=0; j<${#step_sizes[@]}; j++)); do
        param="${param_names[i]}" # Access the element at index i
        step_size="${step_sizes[j]}" # Access the element at index j
        echo $param
        echo -$step_size
        # Generate job name with index
        job_name="computeN_DESY3_"$param"_$step_size"
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
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4

conda init
conda activate massfunction

python -u compute_cluster_abundance_at_cosmology_DESY3.py "$param" -$step_size


EOF
done
done
