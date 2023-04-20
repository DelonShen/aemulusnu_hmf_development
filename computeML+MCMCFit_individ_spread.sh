#!/bin/bash

# Set the data file names
data_files=("Box0_1400" "Box9_1400" "Box19_1400" "Box29_1400" "Box39_1400" "Box49_1400" "Box59_1400" "Box69_1400" "Box79_1400" "Box89_1400" "Box99_1400" "Box_n50_0_1400" "Box_n50_9_1400" "Box_n50_19_1400" "Box_n50_29_1400" "Box_n50_39_1400" "Box_n50_49_1400")

# Set the Slurm job parameters
job_name_prefix="ml+mcmc-fit-sparse"
output_dir="logs"
time_limit="1440:00"
partition="kipac"
num_nodes=1
mem_per_node=8192
cpus_per_task=32

# Loop through each data file
for data_file in "${data_files[@]}"; do
  # Create a Slurm job script for the data file
  job_name="${job_name_prefix}-${data_file}"
  job_script="${job_name}.sh"
  output_file="${output_dir}/$(date +%Y-%m-%d)-${job_name}.out"
  error_file="${output_dir}/$(date +%Y-%m-%d)-${job_name}.err"

  cat > "${job_script}" << EOL
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${output_file}
#SBATCH --error=${error_file}
#SBATCH --time=${time_limit}
#SBATCH -p ${partition}
#SBATCH --nodes=${num_nodes}
#SBATCH --mem=${mem_per_node}
#SBATCH --cpus-per-task=${cpus_per_task}

conda init
conda activate massfunction

python computeMLFit.py ${data_file}
python computeML-MCMC-fit.py ${data_file}
EOL

  # Submit the Slurm job
  sbatch "${job_script}"

  echo "Job submitted for ${data_file}"
done

echo "All jobs submitted."