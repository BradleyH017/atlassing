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

# Set your cluster submission parameters
LSF_PARAMS="-o sm_logs/raw_qc-%J-output.log -e sm_logs/raw_qc-%J-error.log -q gpu-huge -gpu - -G team152 -n 4 -M 400000 -a 'memlimit=True' -R 'select[mem>400000] rusage[mem=400000] span[hosts=1]'"
JOBS=10

# Submit the Snakefile to the cluster using snakemake
source ~/.bashrc
conda activate scvi-env
snakemake -p \
    --snakefile Snakefile \
    --cluster "bsub $LSF_PARAMS" \
    --jobs $JOBS \
    --conda-frontend conda \
    --use-conda \
    /lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/tissues_combined/{rectum,ti,blood}/objects/adata_PCAd_batched.h5ad

# conda activate scvi-env # Must be in an environment with mamba installed
# bsub -M 10000 -a "memlimit=True" -R "select[mem>10000] rusage[mem=10000] span[hosts=1]" -o sm_logs/snakemake_master-%J-output.log -e sm_logs/snakemake_master-%J-error.log -q oversubscribed -J "snakemake_master" < submit_snakemake.sh 

