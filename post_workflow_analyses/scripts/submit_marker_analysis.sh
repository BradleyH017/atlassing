#!/usr/bin/env bash

# Set up dir
repo_dir="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing"
cd $repo_dir

# Activate env
source activate scvi-env

# Get lineage
lineage_list=post_workflow_analyses/scripts/major_lineages.txt
lineage=$(head $lineage_list -n ${LSB_JOBINDEX} | tail -n 1)
echo "RUNNING ON" ${lineage}

# Run script on this lineage
python  post_workflow_analyses/scripts/999-calc_marker_genes.py \
    --input_file "results_round2/combined/objects/adata_grouped_post_cluster_QC.h5ad" \
    --tissue "combined" \
    --subset "manual_lineage" \
    --subset_value ${lineage} \
    --cluster_column "leiden" \
    --method "wilcoxon" \
    --filter "True" \
    --min_in_group_fraction 0.5 \
    --max_out_group_fraction 0.3 \
    --min_fold_change 1.5 \
    --baseout "results_round2" \
    --compare_annots "Celltypist:megagut_celltypist_lowerGI+lym_adult_mar24:predicted_labels,Azimuth:predicted.celltype.l1,Azimuth:predicted.celltype.l2,Azimuth:predicted.celltype.l3,Celltypist:cluster-ti-cd_healthy-freeze005_clustered_final:predicted_labels" \
    --external_marker_sets "external_markers/helmsley_23.txt" \
    --external_marker_set_names "helmsley_23" 

# Run
# cd /lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/post_workflow_analyses/scripts
# bsub -o ../logs/calc_markers-%J-%I-output.log -e ../logs/calc_markers-%J-%I-error.log -q teramem -G team152 -n 1 -M 750000 -a "memlimit=True" -R "select[mem>750000] rusage[mem=750000] span[hosts=1]" -J "calculating_markers[1-5]" < submit_marker_analysis.sh 

