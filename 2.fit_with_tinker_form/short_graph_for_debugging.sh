#!/bin/bash
# Set the Slurm job parameters
job_name_prefix="fit-debug"
output_dir="logs"
time_limit="60:00"
partition="kipac"
num_nodes=1
mem_per_node=32768
cpus_per_task=32

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
#SBATCH --mem=${mem_per_node}
#SBATCH --cpus-per-task=${cpus_per_task}

python -u fit_box.py Box_n50_33_1400 Box_n50_0_1400
python -u fit_box.py Box_n50_42_1400 Box_n50_33_1400
python -u fit_box.py Box_n50_9_1400 Box_n50_42_1400
python -u fit_box.py Box69_1400 Box_n50_9_1400

EOF
