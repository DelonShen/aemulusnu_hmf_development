import subprocess
from tqdm import trange
from datetime import date

param_names = ['10^9 As', 'ns', 'H0', 'w0', 'ombh2', 'omch2']
param_names = ['nu_mass_ev']
step_sizes = [i for i in (float(j) / 100 for j in range(68, 201))]

for i in trange(len(param_names)):
    for j in trange(len(step_sizes)):
        param = param_names[i]
        step_size = step_sizes[j]
        job_name = f"computeN_{param}_{step_size}"
        output_log = f"logs/{date.today().strftime('%Y-%m-%d')}-{job_name}.out"
        error_log = f"logs/{date.today().strftime('%Y-%m-%d')}-{job_name}.err"
        line = f'python -u compute_cluster_abundance_at_cosmology.py "{param}" -{step_size} &> {output_log} 2> {error_log}'
        result = subprocess.run(line, shell=True, check=True)
