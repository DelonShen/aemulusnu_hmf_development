box = 'Box_n50_0_1400'
from utils import *
from massfunction import *

import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os
import emcee
import sys
import numpy as np
import pickle

cosmos_f = open('data/cosmo_params.pkl', 'rb')
cosmo_params = pickle.load(cosmos_f) #cosmo_params is a dict
cosmos_f.close()

cosmo = cosmo_params[box]
mass_function = MassFunction(cosmo)

h = cosmo['H0']/100

NvM_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_NvsM.pkl'
NvM_f = open(NvM_fname, 'rb')
NvMs = pickle.load(NvM_f) #NvMs is a dictionary of dictionaries
NvM_f.close()

a_list = list(reversed(NvMs.keys()))
print(a_list)

import subprocess
from datetime import date


# Submit each command as a Slurm job with the dynamically generated log paths    
for i in trange(1, len(a_list)):
    line = 'python -u fit_individ_iter.py %s %f %s %f'%(box, a_list[i], box, a_list[i-1])
    # Define the job parameters
    job_name = 'fit_%s_a%.2f'%(box, a_list[i])

    # Get the current date
    current_date = date.today().strftime('%Y-%m-%d')

    # Construct the output and error log paths
    output_log = f'logs/{current_date}-{job_name}.out'
    error_log = f'logs/{current_date}-{job_name}.err'
    sbatch_command = f'sbatch --job-name={job_name} --output={output_log} --error={error_log} --time=5:00 -p kipac --nodes=1 --mem=32768 --cpus-per-task=1 --wrap="{line}"'

    
    subprocess.run(sbatch_command, shell=True)

    # Wait for the current job to finish
    while True:
        squeue_output = subprocess.check_output(['squeue', '-u', 'delon', '-n', job_name, '-h']).decode().strip()
        if not squeue_output:
            break