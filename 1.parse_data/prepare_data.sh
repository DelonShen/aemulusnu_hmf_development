#!/bin/bash

# Set data files and Slurm job parameters here
data_files=("Box0_1400" "Box1_1400" "Box2_1400" "Box3_1400" "Box4_1400" "Box5_1400" "Box6_1400" "Box7_1400" "Box8_1400" "Box9_1400" "Box10_1400" "Box11_1400" "Box12_1400" "Box13_1400" "Box14_1400" "Box15_1400" "Box16_1400" "Box17_1400" "Box18_1400" "Box19_1400" "Box20_1400" "Box21_1400" "Box22_1400" "Box23_1400" "Box24_1400" "Box25_1400" "Box26_1400" "Box27_1400" "Box28_1400" "Box29_1400" "Box30_1400" "Box31_1400" "Box32_1400" "Box33_1400" "Box34_1400" "Box35_1400" "Box36_1400" "Box37_1400" "Box38_1400" "Box39_1400" "Box40_1400" "Box41_1400" "Box42_1400" "Box43_1400" "Box44_1400" "Box45_1400" "Box46_1400" "Box47_1400" "Box48_1400" "Box49_1400" "Box50_1400" "Box51_1400" "Box52_1400" "Box53_1400" "Box54_1400" "Box55_1400" "Box56_1400" "Box57_1400" "Box58_1400" "Box59_1400" "Box60_1400" "Box61_1400" "Box62_1400" "Box63_1400" "Box64_1400" "Box65_1400" "Box66_1400" "Box67_1400" "Box68_1400" "Box69_1400" "Box70_1400" "Box71_1400" "Box72_1400" "Box73_1400" "Box74_1400" "Box75_1400" "Box76_1400" "Box77_1400" "Box78_1400" "Box79_1400" "Box80_1400" "Box81_1400" "Box82_1400" "Box83_1400" "Box84_1400" "Box85_1400" "Box86_1400" "Box87_1400" "Box88_1400" "Box89_1400" "Box90_1400" "Box91_1400" "Box92_1400" "Box93_1400" "Box94_1400" "Box95_1400" "Box96_1400" "Box97_1400" "Box98_1400" "Box99_1400" "Box_n50_0_1400" "Box_n50_1_1400" "Box_n50_2_1400" "Box_n50_3_1400" "Box_n50_4_1400" "Box_n50_5_1400" "Box_n50_6_1400" "Box_n50_7_1400" "Box_n50_8_1400" "Box_n50_9_1400" "Box_n50_10_1400" "Box_n50_11_1400" "Box_n50_12_1400" "Box_n50_13_1400" "Box_n50_14_1400" "Box_n50_15_1400" "Box_n50_16_1400" "Box_n50_17_1400" "Box_n50_18_1400" "Box_n50_19_1400" "Box_n50_20_1400" "Box_n50_21_1400" "Box_n50_22_1400" "Box_n50_23_1400" "Box_n50_24_1400" "Box_n50_25_1400" "Box_n50_26_1400" "Box_n50_27_1400" "Box_n50_28_1400" "Box_n50_29_1400" "Box_n50_30_1400" "Box_n50_31_1400" "Box_n50_32_1400" "Box_n50_33_1400" "Box_n50_34_1400" "Box_n50_35_1400" "Box_n50_36_1400" "Box_n50_37_1400" "Box_n50_38_1400" "Box_n50_39_1400" "Box_n50_40_1400" "Box_n50_41_1400" "Box_n50_42_1400" "Box_n50_43_1400" "Box_n50_44_1400" "Box_n50_45_1400" "Box_n50_46_1400" "Box_n50_47_1400" "Box_n50_48_1400" "Box_n50_49_1400")
#data_files=("Box_n50_1_1400")

job_name_prefix="prepare-data"
output_dir="logs"
time_limit="1440:00"
partition="kipac"
num_nodes=1
mem_per_node=8192
cpus_per_task=1

# Loop through each data file
for data_file in "${data_files[@]}"; do
  # Set the job name, output file, and error file
  job_name="${job_name_prefix}-${data_file}"
  output_file="${output_dir}/$(date +%Y-%m-%d)-${job_name}.out"
  error_file="${output_dir}/$(date +%Y-%m-%d)-${job_name}.err"

  # Submit the Slurm job using heredoc
  sbatch << EOF
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
python -u prepare_data.py ${data_file} ${SUFFIX}
EOF

  echo "Job submitted for ${data_file}"
done

echo "All jobs submitted."
