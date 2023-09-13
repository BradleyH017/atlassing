#! /bin/bash
#BSUB -o logs/preprocessing-%J-output.log
#BSUB -e logs/preprocessing-%J-error.log 
#BSUB -q normal
#BSUB -G team152
#BSUB -n 1
#BSUB -M 10000
#BSUB -a "memlimit=True"
#BSUB -R "select[mem>10000] rusage[mem=10000] span[hosts=1]"

# Make sure am in snakemake environment
cd /lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/workflow
snakemake /lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/rectum/healthy/All_not_gt_subset/figures/embedding_param_sweep/umap500nn_2_min_dist_3_spread.pdf --cores 1
# bsub -o submit_logs/atlas-%J-output.log -e submit_logs/atlas-%J-error.log -q long -J "snakemake_atlas" < snakemake_atlas_submit.sh
