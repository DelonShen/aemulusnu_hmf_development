import subprocess
from tqdm import trange, tqdm
from datetime import date

# param_names = ['10^9 As', 'ns', 'H0', 'w0', 'ombh2', 'omch2']
# param_names = ['nu_mass_ev']
# step_sizes = [i for i in (float(j) / 100 for j in range(68, 201))]



fiducial_log10_rel_step_size = { #for numerical derivativese
    '10^9 As': -2.6,
    'ns': -4,
    'H0': -2.3,
    'w0': -2.3,
    'ombh2': -2.2,
    'omch2': -2.6,
    'nu_mass_ev': -1.08,
}

for key in tqdm(fiducial_log10_rel_step_size):
    param = key
    step_size = -fiducial_log10_rel_step_size[key]
    job_name = f"computeN_{param}_{step_size}"
    output_log = f"logs/{date.today().strftime('%Y-%m-%d')}-{job_name}.out"
    error_log = f"logs/{date.today().strftime('%Y-%m-%d')}-{job_name}.err"
    line = f'python -u compute_cluster_abundance_at_cosmology.py "{param}" -{step_size}'
    print(line)
    result = subprocess.run(line, shell=True, check=True)
