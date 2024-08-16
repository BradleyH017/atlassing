#!/bin/bash
#BSUB -o sm_logs/snakemake_master-%J-output.log
#BSUB -e sm_logs/snakemake_master-%J-error.log 
#BSUB -q oversubscribed
#BSUB -G team152
#BSUB -n 1
#BSUB -M 10000
#BSUB -a "memlimit=True"
#BSUB -R "select[mem>10000] rusage[mem=10000] span[hosts=1]"
#BSUB -J 1

## Getting LSF snakemake config:
# mamba install -c conda-forge cookiecutter
# profile_dir="${HOME}/.config/snakemake"
# mkdir -p "$profile_dir"
## use cookiecutter to create the profile in the config directory
# template="gh:Snakemake-Profiles/lsf"
# cookiecutter --output-dir "$profile_dir" "$template"
# LSF_UNIT_FOR_LIMITS=2 (MB)
# UNKWN_behaviour=2 (kill)
# ZOMBI_behaviour=2 (kill)
# latency wait = 5
# use_conda = y
# use singularity = n
# restart_times = 0
# print shell commands = y
# jobs 500
# default mem = 1024
# default cluster logdir = ./ (tissues_combined/sm_logs)
# default queue = normal
# default project = ""
# max_status_checks_per_second = 10
# max_jobs_per_second = 10
# max_status_checks = 1
# wait_between_tries = 1
# jobscript timeout 10
# profile_name = sanger-brad2

# Define some params
config_var=configs/config_cluster_within_lineage.yaml # Using the within lineage config (for round 2 analysis)
worfklow_prefix="multi-tissue_"
group="team152"
workdir=${PWD}

# Submit the Snakefile to the cluster using snakemake
#source ~/.bashrc # activate base env - farm5
module load HGI/common/conda
#conda init
source activate sm7


#conda activate scvi-env
# module load HGI/softpack/groups/hgi/snakemake/7.32.4
#module load ISG/singularity/3.9.0 # farm5
module load ISG/singularity/3.11.4 # farm22

# Modify the name of the results directory (from the previous run) - so that we can create a new directory for this run
if [ ! -d "results_round1" ]; then
    mv results results_round1
    mv sm_logs sm_logs_round1   
    mkdir -p sm_logs 
fi

# Copy config to results
mkdir -p results
cp $config_var results/

# Build dag
snakemake -j 50 \
    --latency-wait 90 \
    --rerun-incomplete \
    --keep-incomplete \
    --keep-going \
    --default-resources threads=1 mem_mb=2000 \
    --directory ${workdir} \
    --use-conda \
    --conda-frontend conda \
    --cluster-config ${config_var} \
    --use-singularity \
    --singularity-args "-B /lustre -B /software" \
    --cluster " mkdir -p 'sm_logs/cluster/${worfklow_prefix}_{rule}'; bsub -q {resources.queue} -R 'rusage[mem={resources.mem_mb}] select[mem>{resources.mem_mb}] span[hosts=1]' -M {resources.mem_mb} -n {resources.threads} -J '${worfklow_prefix}_{rule}.{wildcards}' -G ${group} -o 'sm_logs/cluster/${worfklow_prefix}_{rule}/{rule}.{wildcards}.%J-out' -e 'sm_logs/cluster/${worfklow_prefix}_{rule}/{rule}.{wildcards}.%J-err'" \
    -s workflows/Snakefile_cluster_within_lineage-002.smk \
    --until all \
    --dag --rerun-triggers mtime | dot -Tpng > dags/dag_cluster_within.png

# Execute script (updating config params to use optimum model params)
snakemake -j 50 \
    --latency-wait 90 \
    --rerun-incomplete \
    --keep-incomplete \
    --keep-going \
    --default-resources threads=1 mem_mb=2000 \
    --directory ${workdir} \
    --use-conda \
    --conda-frontend conda \
    --cluster-config ${config_var} \
    --use-singularity \
    --singularity-args "-B /lustre -B /software" \
    --cluster " mkdir -p 'sm_logs/cluster/${worfklow_prefix}_{rule}'; bsub -q {resources.queue} -R 'rusage[mem={resources.mem_mb}] select[mem>{resources.mem_mb}] span[hosts=1]' -M {resources.mem_mb} -n {resources.threads} -J '${worfklow_prefix}_{rule}.{wildcards}' -G ${group} -o 'sm_logs/cluster/${worfklow_prefix}_{rule}/{rule}.{wildcards}.%J-out' -e 'sm_logs/cluster/${worfklow_prefix}_{rule}/{rule}.{wildcards}.%J-err'" \
    -s workflows/Snakefile_cluster_within_lineage-002.smk \
    --rerun-triggers mtime \
    --until all 

# NOTE: Have adjusted to run original model to test
# Add the following to overwrite with optimum params
# --config optimise_run_params=False sparsity_l1__activity=0.01 sparsity_l1__bias=0.0001 sparsity_l1__kernel=0.0001 sparsity_l2__activity=0.0001 sparsity_l2__bias=0.01 sparsity_l2__kernel=0.01 \


# bsub -M 2000 -a "memlimit=True" -R "select[mem>2000] rusage[mem=2000] span[hosts=1]" -o sm_logs/snakemake_master-%J-output.log -e sm_logs/snakemake_master-%J-error.log -q oversubscribed -J "snakemake_master" < submit_snakemake_cluster_in_lineage-002.sh 


