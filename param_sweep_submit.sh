#! /bin/bash
#BSUB -o logs/preprocessing-%J-output.log
#BSUB -e logs/preprocessing-%J-error.log 
#BSUB -q normal
#BSUB -G team152
#BSUB -n 4
#BSUB -M 300000
#BSUB -a "memlimit=True"
#BSUB -R "select[mem>300000] rusage[mem=300000] span[hosts=1]"
#BSUB -J 9

# Define option
cd /lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis
echo $LSB_JOBINDEX

# For the sweep of parameters
NN_list=params_lists/NN_opt.txt
NN_opt=$(head $NN_list -n ${LSB_JOBINDEX} | tail -n 1)
echo "The option for NN is"
echo $NN_opt
echo "Running python script"
python scripts/scRNAseq/rectum_embedding_param_sweep.py $NN_opt
#bsub -o logs/preprocessing-%J-%I-output.log -e logs/preprocessing-%J-%I-error.log -q long -J "rectum_param_sweep[1-9]" < python_submit_script.sh 
