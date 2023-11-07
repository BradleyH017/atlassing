#! /bin/bash
#BSUB -o logs/preprocessing-%J-output.log
#BSUB -e logs/preprocessing-%J-error.log 
#BSUB -q normal
#BSUB -G team152
#BSUB -n 6
#BSUB -M 200000
#BSUB -a "memlimit=True"
#BSUB -R "select[mem>200000] rusage[mem=200000] span[hosts=1]"
#BSUB -J 9

# Define option
cd /lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis

# To test an array of cluster resolution values and predicting annotations using keras
cd scripts/scRNAseq/Atlassing/
python clustering_array.py

# Running on GPUs
# bsub -o logs/keras_clustering-%J-output.log -e logs/keras_clustering-%J-error.log -q gpu-huge -gpu - -J "keras_clustering" < cluster_sweep_submit.sh 

