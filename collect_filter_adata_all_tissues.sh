#!/bin/bash
#BSUB -o sm_logs/atlassing_prep-%J-output.log
#BSUB -e sm_logs/atlassing_prep-%J-error.log 
#BSUB -q teramem
#BSUB -G team152
#BSUB -n 4
#BSUB -M 950000
#BSUB -a "memlimit=True"
#BSUB -R "select[mem>950000] rusage[mem=950000] span[hosts=1]"
#BSUB -J 1


# Submit script
python bin/000_gather_probabilities_adatas_v7.py
# bsub -o sm_logs/atlassing_prep-%J-output.log -e sm_logs/atlassing_prep-%J-error.log -q teramem -J "atlassing_prep" < collect_filter_adata_all_tissues.sh 

