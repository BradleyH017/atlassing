#! /bin/bash
#BSUB -o logs/preprocessing-%J-output.log
#BSUB -e logs/preprocessing-%J-error.log 
#BSUB -q normal
#BSUB -G team152
#BSUB -n 30
#BSUB -M 200000
#BSUB -a "memlimit=True"
#BSUB -R "select[mem>200000] rusage[mem=200000] span[hosts=1]"
#BSUB -J 9

# Define option
cd /lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis

## FOR THE ATLASSING OF THE RECTUM (#BSUB -M 200000)
# This is done on the 'All_not_gt_subset' data
# Use the sc4 environment
cd scripts/scRNAseq/Atlassing/
python atlassing_rectum_all_samples.py
# bsub -o logs/rectum_atlassing01-%J-output.log -e logs/rectum_atlassing01-%J-error.log -q long -J "rectum_atlassing01" < atlassing_rectum_submit.sh 


# Running on GPUs
# bsub -o logs/rectum_atlassing01-%J-output.log -e logs/rectum_atlassing01-%J-error.log -q gpu-huge -gpu - -J "rectum_atlassing01" < atlassing_rectum_submit.sh 

