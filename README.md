# Dynamic, multi-tissue QC and clustering
### Author: Bradley Harris
This repo contains code and workflows to run the multi-round atlassing of the entire IBDverse single cell dataset.
All executables and custom scripts are present within the `bin` directory

# Replication
To reproduce all analysis, simply download the input file (LINK TO COME), and execute each workflow in order. Details of which are provided below.

## Round 1
First, hard thresholds are applied to the entire dataset as a whole with the goal of identifying broad cell clusters (major lineages). This is done by execution of `workflows/Snakefile`. For submission to an LSF cluster, see `submit_snakemake.sh`.

#### Options
This workflow uses the options present in `configs/config.yaml`. Outputs are saved in the 'results' directory produced during the worfklow. Configuration of input files and names at the top of this workflow mean it could be ran on multiple datasets side by side.

#### End summary
As this is the first step, it is summarised by `bin/990-round1_round2_bridging_script.py`. This annotates the anndata object and saves new objects into `input_cluster_within`, that are used as input for the next round.

## Round 2
After hard thresholds, each major lineage is taken and subject to relative QC by `workflows/Snakefile_cluster_within_lineage-002.smk`. Dynamic thresholds (median absolute deviations - MAD) are appllied to MT%, log10(counts) and log10(nGenes), as well as sample-level QC. This requires some manual reconstruction of the directory structure, which is handled in the LSF-submission script `submit_snakemake_cluster_in_lineage-002.sh` to submit `workflows/Snakefile_cluster_within_lineage-002.smk`.

#### Options
All options, such as directionality of relative thresholds, strictness and method for their application are found in `configs/config_cluster_within_lineage.yaml`

#### End summary
While this workflow does suggest optimum resolutions after some cluster QC, it is heavily recommended that authors conduct their own thorough QC of clusters. Hence, this workflow is followed by `bin/990-round2_round3_bridging_script.py` where clustering resolutions and excluded clusters can be manually selected.

## Round 3
The high QC data is then re-integrated for two reasons; 1) Generation of a complete dataset embedding and 2) Generate autoannotations models across the entire dataset that an be applied to the remaining and incoming data. This is performed by `workflows/Snakefile_re_embed_after_within_lineage-003.smk`.

#### Options
Options for this workflow are stored in `configs/config_re_embed_after_within_lineage.yaml`, but should broadly follow that of the within clustering **as no additional cells should be removed by this workflow**. This should therefore not utilise the relative thresholding method by specifying `threshold_method: "None"`. The major choice here is whether or not to generate a complete-dataset `Celltypist` model. A keras model for major lineages at the entire dataset level can also be calculated.

## Round 4
If keras autoannotation models are being used, then round 4 can be executed by running `workflows/Snakefile_predict_all_cells-004.smk`.

### Options
Defining options here, specificied in `configs/config_predict_all_cells.yaml` are the anndata object to predict all cells from (`original_input_round1`) and the threshold (probability_threshold).
