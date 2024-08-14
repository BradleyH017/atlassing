#!/usr/bin/env python
# Bradley August 2024
# Script to manually recombine the per-lineage input data for round 3 input

# Import libraries
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os

# Define lineages, desired resolutions and clusters to chuck per lineage
lins_res = {"all_Epithelial": 1.1, "all_Mesenchymal": 0.7, "all_Myeloid": 0.7, "all_B": 0.6, "all_T": 1.0, "Platelet+RBC": np.nan, "Mast": np.nan}
chuck = {"all_Epithelial": ["6", "27"], "all_Mesenchymal": [None], "all_Myeloid": [None], "all_B": ["14", "13"], "all_T": [None]}

# Load in the combined output from round1
adata = sc.read_h5ad("results_round1/all/objects/adata_manual_lineage_clean.h5ad")

# Read in the final clusters
final_clusters = []
for l in lins_res:
    print(f"Working on: {l}")
    res = lins_res[l]
    if not np.isnan(res):
        clusters = pd.read_csv(f"results/{l}/tables/clustering_array/leiden_{str(res)}/clusters.csv")
        clusters.rename(columns={f"leiden_{res}": "leiden"}, inplace=True)
        clusters = clusters[~clusters['leiden'].astype(str).isin(chuck[l])]
        pref = l.replace("all_", "")
        clusters['leiden'] = pref + "_" + clusters['leiden'].astype(str)
    else:
        clusters = pd.DataFrame({"cell": adata.obs[adata.obs['manual_lineage'] == l].index.values, "leiden": l})
    
    final_clusters.append(clusters)

final_clusters = pd.concat(final_clusters)

# Merge, and subset the adata to include only these cells
adata.obs.reset_index(inplace=True)
adata = adata[adata.obs['cell'].isin(final_clusters['cell'])]
adata.obs = adata.obs.merge(final_clusters, on="cell", how="left")
adata.obs.set_index("cell", inplace=True)
        
# Save this so it can be passed to round 3
out="results/combined/objects"
if os.path.exists(out) == False:
        os.makedirs(out)

adata.write_h5ad("results/combined/objects/adata_grouped_post_cluster_QC.h5ad")

# Also save a list of the clusters to chuck (convenient when predicting)
keys = []
values = []
for key, value_list in chuck.items():
    for value in value_list:
        keys.append(key)
        values.append(value)

# Create the DataFrame
df = pd.DataFrame({
    'manual_lineage': keys,
    'chuck': values
})
df.to_csv("results/combined/objects/chuck_clusters.csv")