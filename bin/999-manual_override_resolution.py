#### Bradley July 2024 ####
# Manually override the optim_resolution selected by keras post cluster QC. Also decide whether there should be blacklisted samples or not
# NOTE: This is to be executed following some analysis of the MCC results per lineage
# This script will overwrite the anndata used to combine together, make the correlated black list samples dataframe, and exclude desired clusters from the analysis if desired

# Import packages
import scanpy as sc
import numpy as np
import pandas as pd
import os
from numpy import savetxt
from numpy import asarray

# Define params
tissue = "all_epithelial"
use_res = 0.75 # Manually-defined resolution
max_proportion = 0.2 # Max proportion of cluster cells from single sample. This does not necessarily need to match the config file. However, it will be used to redefine the blacklist samples which may/may not be used to filter samples when combining adatas (006b-combine_across_lineages)
MCC_thresh = 0 # This does not need to match that of the config
remove_bad_cluster = "sample" # Remove bad cluster on basis of what? If 'sample' bad cluster will be removed due to blacklist sample. Could also be MCC (NOT DONE) or both


os.chdir("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined")

# override optim res
savetxt(f"results/{tissue}/tables/optim_resolution.txt", asarray([[float(use_res)]]), delimiter='\t')

# import adata for manual res
adata = sc.read_h5ad(f"results/{tissue}/tables/clustering_array/leiden_{use_res}/adata_PCAd_batched_umap_{use_res}.h5ad")

# Redefine the blacklist samples
proportions = adata.obs.groupby('leiden')['samp_tissue'].value_counts(normalize=True).reset_index(name="samp_tissue_proportion")
max_index = proportions.groupby('leiden')["samp_tissue_proportion"].idxmax()
prop_sum = proportions.loc[max_index]
bad_samples = prop_sum.loc[prop_sum['samp_tissue_proportion'] > max_proportion, "samp_tissue"].values
black_list_df = pd.DataFrame({"res": 'leiden', "samp_tissue": bad_samples})
bl_samp_counts = adata.obs.loc[adata.obs['samp_tissue'].isin(bad_samples), 'samp_tissue'].value_counts().reset_index()
black_list_df = black_list_df.merge(bl_samp_counts, on="samp_tissue", how="left")
black_list_df['res'] = black_list_df['res'] + "_" + str(use_res)
black_list_df.to_csv(f"results/{tissue}/figures/clustering_array_summary/black_list_samples_postQC.csv")

# Save a file to indicate the override has taken place
savetxt(f"results/{tissue}/tables/OVERRIDE_optim_resolution.txt", asarray([[float(1)]]), delimiter='\t')

# Remove cluster / samples
if remove_bad_cluster == "sample":
    bad_cluster = prop_sum[ ( prop_sum['samp_tissue_proportion'] > max_proportion ) & ( prop_sum['samp_tissue'].isin(bad_samples) )]
    bad_cluster = bad_cluster['leiden'].values[0]
    adata = adata[~(adata.obs['leiden'] == bad_cluster)]
    
# OVERIDE the adata
prefix = tissue.split("_")[1]
adata.obs['leiden'] = prefix + "_" + adata.obs['leiden'].astype(str)
adata.write(f"results/{tissue}/objects/adata_clusters_post_clusterQC.h5ad")

# If the combination script has been ran, then delete the output from this to ensure re-derivation when the pipeline is re-ran
combi = "results/combined/objects/adata_grouped_post_cluster_QC.h5ad"
if os.path.exists(combi):
    os.remove(combi)