#!/bin/bash

# Get the list of job IDs containing "compute-ml-fit"
job_ids=$(squeue -u delon -o "%i %j" | grep "ml+mcmc" | awk '{print $1}')

# Check if any jobs match the criteria
if [[ -z $job_ids ]]; then
  echo "No jobs found with the specified criteria."
  exit 0
fi

# Cancel each job using scancel
for job_id in $job_ids; do
  echo "Cancelling job: $job_id"
  scancel $job_id
done

echo "All jobs with names containing 'compute-ml-fit' have been cancelled."

exit 0
