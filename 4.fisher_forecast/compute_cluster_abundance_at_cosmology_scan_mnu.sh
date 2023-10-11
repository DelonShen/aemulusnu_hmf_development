#!/bin/bash


param_names=('10^9 As' 'omch2')
step_sizes=(2.6 2.3)

#from PDG
#Delta m21^2 = 7.53   e-5  eV^2
#Delta m32^2 = -2.519 e-3  eV^2 (inverted)
#Delta m32^2 = 2.437  e-3  eV^2 (normal)
#Assume normal ordering and m1 = 0 
#This means that 
# m2 ~ .008 eV
# m3 ~ .06  eV
# \sum mnu ~ .06 eV
# I think federico claims that
# considering A_lens, we have the Planck PR3 constraint \sum mnu < .286
# So lets scan from mnu = .06 -> .3?

mnus=()
for i in $(seq 0.06 0.01 .302); do
  mnus+=($i)
done



for((i=0; i<${#mnus[@]}; i++)); do
    for ((j=0; j<${#param_names[@]}; j++)); do
        mnu="${mnus[i]}"
        param="${param_names[j]}" 
        step_size="${step_sizes[j]}" 
        echo $param
        echo -$step_size
        echo $mnu
        # Generate job name with index
        job_name="computeN_"$param"_"$step_size"_$mnu"
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
#SBATCH --time=40:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=4096
#SBATCH --cpus-per-task=16

conda init
conda activate massfunction

python -u compute_cluster_abundance_at_cosmology.py "$param" -$step_size $mnu


EOF
done
done
