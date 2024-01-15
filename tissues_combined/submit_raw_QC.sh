#! /bin/bash
#BSUB -o logs/scANVI_ref-%J-output.log
#BSUB -e logs/scANVI_ref-%J-error.log 
#BSUB -q normal
#BSUB -G team152
#BSUB -n 4
#BSUB -M 450000
#BSUB -a "memlimit=True"
#BSUB -R "select[mem>450000] rusage[mem=450000] span[hosts=1]"
#BSUB -J 9

# Define run options
blood_file="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/yascp_analysis/blood/results/merged_h5ad/outlier_filtered_adata.h5ad"
TI_file="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/anderson_ti_freeze003_004-otar-processed.h5ad"
rectum_file="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/2023_09_rectum/adata.h5ad"
outdir="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/tissues_combined"
discard_other_inflams="yes" # Remove other inflammations and UC?
all_blood_immune="yes" # Treat all blood cells as immune?
min_nUMI=200 # min UMI per cell
use_absolute_nUMI="no" # Filter on the basis of absolute nUMI per cell
relative_grouping="tissue|lineage" # How to define relative thresholds? e.g "tissue|lineage" will define relative thresholds within each lineage or each tissue
relative_nMAD_threshold=2.5 # Cut off for relative thresholds (per lineage per tissue)
min_nGene=400 # Absolute cut off for nGene per cell
use_absolute_nGene="no" # Use absolute?
MTgut=50 # MT% cutt off for gut
MTblood=20 # MT% cutt off for blood
use_absolute_MT="no" # Use absolute?
absolute_max_MT=50 # Absolute max MT% irrespective of relative cut off
min_mean_nCount_per_samp_blood=3000 # Absolute thresh (sample level)
min_mean_nCount_per_samp_gut=7500 # Absolute thresh (sample level)
min_mean_nGene_per_samp_blood=1000 # Absolute thresh (sample level)
min_mean_nGene_per_samp_gut=1750 # Absolute thresh (sample level)
use_abs_per_samp="no" # Use absolute?
filt_blood_keras="no" # Filter blood for keras annotation
n_variable_genes=4000 # N variable genes
remove_problem_genes="yes" # Remove HLA|RP|MT?
relative_cutoff_within="lineage" 

# Execute
python tissues_combined/001_raw_QC.py \
    --blood_file $blood_file \
    --TI_file $TI_file \
    --rectum_file $rectum_file \
    --outdir $outdir \
    --discard_other_inflams $discard_other_inflams \
    --all_blood_immune $all_blood_immune \
    --min_nUMI $min_nUMI \
    --use_absolute_nUMI $use_absolute_nUMI \
    --relative_grouping $relative_grouping \
    --relative_nMAD_threshold $relative_nMAD_threshold \
    --min_nGene $min_nGene \
    --use_absolute_nGene $use_absolute_nGene \
    --MT_thresh_gut $MTgut \
    --MT_thresh_blood $MTblood \
    --use_absolute_MT $use_absolute_MT \
    --absolute_max_MT $absolute_max_MT \
    --min_mean_nCount_per_samp_blood $min_mean_nCount_per_samp_blood \
    --min_mean_nCount_per_samp_gut $min_mean_nCount_per_samp_gut \
    --min_mean_nGene_per_samp_blood $min_mean_nGene_per_samp_blood \
    --min_mean_nGene_per_samp_gut $min_mean_nGene_per_samp_gut \
    --use_abs_per_samp $use_abs_per_samp \
    --filt_blood_keras $filt_blood_keras \
    --n_variable_genes $n_variable_genes \
    --remove_problem_genes $remove_problem_genes 

# conda activate scvi-env 
# bsub -o logs/qc_batch_all_tissues-%J-output.log -e logs/qc_batch_all_tissues-%J-error.log -q gpu-huge -gpu - -J "qc_batch_all_tissues" < tissues_combined/submit_raw_QC.sh 



