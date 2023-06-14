import sys

box_prev =  sys.argv[2]
box = sys.argv[1]
print('Curr: %-10s, Prev: %-10s'%(box, box_prev))
import numpy as np
from tqdm import tqdm, trange
import os
import numpy as np
import pickle


a_list_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/alist.pkl'
a_list_f = open(a_list_fname, 'rb')
a_list = pickle.load(a_list_f) 
a_list_f.close()
print('alist', a_list)

import subprocess
from datetime import date



line = 'python -u fit_individ_iter.py %s %f %s %f'%(box, 1.0, box_prev, 1.0)
print(line)
# Define the job parameters
job_name = 'fit_%s_a%.2f'%(box, 1.0)

# Get the current date
current_date = date.today().strftime('%Y-%m-%d')

# Construct the output and error log paths
output_log = f'logs/{current_date}-{job_name}.out'
error_log = f'logs/{current_date}-{job_name}.err'
sbatch_command = f'sbatch --job-name={job_name} --output={output_log} --error={error_log} --time=60:00 -p kipac --nodes=1 --mem=32768 --cpus-per-task=1 --wrap="{line}"'


subprocess.run(sbatch_command, shell=True)

# Wait for the current job to finish
while True:
    squeue_output = subprocess.check_output(['squeue', '-u', 'delon', '-n', job_name, '-h']).decode().strip()
    if not squeue_output:
        break
        
        
# Submit each command as a Slurm job with the dynamically generated log paths    
for i in trange(1, len(a_list)):
    line = 'python -u fit_individ_iter.py %s %f %s %f'%(box, a_list[i], box, a_list[i-1])
    print(line)
    # Define the job parameters
    job_name = 'fit_%s_a%.2f'%(box, a_list[i])

    # Get the current date
    current_date = date.today().strftime('%Y-%m-%d')

    # Construct the output and error log paths
    output_log = f'logs/{current_date}-{job_name}.out'
    error_log = f'logs/{current_date}-{job_name}.err'
    sbatch_command = f'sbatch --job-name={job_name} --output={output_log} --error={error_log} --time=60:00 -p kipac --nodes=1 --mem=32768 --cpus-per-task=1 --wrap="{line}"'

    
    subprocess.run(sbatch_command, shell=True)

    # Wait for the current job to finish
    while True:
        squeue_output = subprocess.check_output(['squeue', '-u', 'delon', '-n', job_name, '-h']).decode().strip()
        if not squeue_output:
            break