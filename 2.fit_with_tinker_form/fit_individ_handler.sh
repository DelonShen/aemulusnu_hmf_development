#!/bin/bash
curr=$1
prev=$2

# Set the Slurm job parameters
job_name_prefix="fit-iter-handler-"$curr
output_dir="logs"
time_limit="8:00"
partition="kipac"
num_nodes=1
mem_per_cpu=8192
cpus_per_task=4

# Create a Slurm job script for the data file
job_name="${job_name_prefix}"
job_script="scripts/${job_name}.sh"
output_file="${output_dir}/$(date +%Y-%m-%d)-${job_name}.out"
error_file="${output_dir}/$(date +%Y-%m-%d)-${job_name}.err"

# Submit SLURM job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${output_file}
#SBATCH --error=${error_file}
#SBATCH --time=${time_limit}
#SBATCH -p ${partition}
#SBATCH --nodes=${num_nodes}
#SBATCH --mem-per-cpu=${mem_per_cpu}
#SBATCH --cpus-per-task=${cpus_per_task}

python -u fit_individ_iter_handler.py $curr $prev
EOF
