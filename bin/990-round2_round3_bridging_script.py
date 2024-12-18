#!/usr/bin/env python
# Bradley August 2024
# Script to manually recombine the per-lineage input data for round 3 input

# Import libraries
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
chuck_additional = True # Decide whether to chuck the additional chuck clusters or not? (Based on annotation of markers etc)

# Define lineages, desired resolutions and clusters to chuck per lineage
lins_res = {"all_Epithelial": 0.9, "all_Mesenchymal": 1.0, "all_Myeloid": 0.7, "all_B": 0.8, "all_T": 1.1, "Platelet+RBC": np.nan, "Mast": np.nan}
chuck = {"all_Epithelial": ["11", "33", "34"], "all_Mesenchymal": ["15"], "all_Myeloid": ["6", "13", "18"], "all_B": ["14", "15"], "all_T": ["17", "18", "23", "25"]} # Removed on the basis of outlying QC or a maximum contribution by single sample > 40% 
additional_chuck = {"all_Epithelial": ["8", "30", "21"], "all_Mesenchymal": ["8", "10", "12"], "all_B": ["7", "9", "11"], "all_Myeloid": ["12", "9", "5", "15"], "all_T": ["14", "22", "11"]} 
# Additional were removed on the basis of marker gene expression (either not sufficient for given lineage, or cross-lineage).
# Myeloid 5 looked like bin for low quality DC cells and Myeloid 15 had outlying QC


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
        if chuck_additional and l in additional_chuck.keys():
            clusters = clusters[~clusters['leiden'].astype(str).isin(additional_chuck[l])]
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

if chuck_additional:
    for key, value_list in additional_chuck.items():
        for value in value_list:
            keys.append(key)
            values.append(value)

# Create the DataFrame
df = pd.DataFrame({
    'manual_lineage': keys,
    'chuck': values
})
df.to_csv("results/combined/objects/chuck_clusters.csv")