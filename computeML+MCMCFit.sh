#!/bin/bash

# Set the batch size
batch_size=10

# Loop from 0 to 9 to generate the job scripts
for i in {0..9}; do
  # Generate the job script file name
  job_script="job_${i}.sh"

  # Open the job script for writing
  echo "#!/bin/bash" > "${job_script}"
  echo "#SBATCH --job-name=compute-ml-mcmc-fit-${i}" >> "${job_script}"
  echo "#SBATCH --output=logs/$(date +%Y-%m-%d)-compute-ml-mcmc-fit-${i}.out" >> "${job_script}"
  echo "#SBATCH --error=logs/$(date +%Y-%m-%d)-compute-ml-mcmc-fit-${i}.err" >> "${job_script}"
  echo "#SBATCH --time=720:00" >> "${job_script}"
  echo "#SBATCH -p kipac" >> "${job_script}"
  echo "#SBATCH --nodes=1" >> "${job_script}"
  echo "#SBATCH --mem=8192" >> "${job_script}"
  echo "#SBATCH --cpus-per-task=32" >> "${job_script}"
  echo "" >> "${job_script}"
  echo "conda init" >> "${job_script}"
  echo "conda activate massfunction" >> "${job_script}"

  # Loop to generate the Python commands for the batch
  for j in {0..9}; do
    # Calculate the file number
    file_number=$((i * batch_size + j))

    # Generate the Python command
    echo "python computeML-MCMC-fit.py Box${file_number}_1400" >> "${job_script}"
    echo "python computeML-MCMC-fit.py Box${file_number}_1400"
  done

  # Submit the job script to Slurm
  sbatch "${job_script}"

  echo "Job ${i} submitted."
done

echo "All jobs submitted."