#!/usr/bin/env python
####################################################################################################################################
####################################################### Bradley April 2024 #########################################################
# Bridging script to define lineage from the coarse clusters (res=0.1) defined from the first round of analysis ####################
# To be ran after first round of analysis (Snakefile with config.yaml), and before second round (Snakefile with config_in_lineage) #
####################################################################################################################################
####################################################################################################################################

# Packages
import scanpy as sc
import pandas as pd
import numpy as np

# Define tissue and resolution to define lineages from (selection of clusters for each lineage will vary depending on this)
tissue="rectum"
resolution="0.25"

# Import data
fpath = f"results/{tissue}/tables/clustering_array/leiden_{resolution}/adata_PCAd_batched_umap_{resolution}.h5ad"
adata = sc.read_h5ad(fpath)

# Plot with labels on data for ease
sc.settings.figdir=f"results/{tissue}/figures/UMAP/annotation"
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")
sc.pl.umap(adata, color="leiden", legend_loc="on data", save=f"_{tissue}_leiden_{resolution}_legend_on.png")

# Generate maps of clusters --> manual lineages
manual_mapping = {"0": "epithelial","1": "epithelial", "2": "epithelial", "3": "immune", "4": "immune", "5": "immune","6": "epithelial", "7": "epithelial", "8": "immune", "9": "epithelial", "10": "epithelial", "11": "immune", "12": "epithelial", "13": "mesenchymal", "14": "immune", "15": "epithelial", "16": "immune", "17": "epithelial"}


# Add to dataframe and rename the column
adata.obs['manual_lineage'] = adata.obs['leiden'].map(manual_mapping)
adata.obs = adata.obs.rename(columns={"leiden": f"leiden_{resolution}_round1_QC"}) 
if "cluster" in adata.obs.columns:
    adata.obs.drop(columns="cluster", inplace=True)

# Also remove the previous keep/remove columns
columns_to_remove = [col for col in adata.obs.columns if 'keep' in col]
adata.obs.drop(columns=columns_to_remove, inplace=True)

# Plot the manual lineage
sc.pl.umap(adata, color="manual_lineage", legend_loc="on data", save=f"_{tissue}_manual_lineage_legend_on.png")

# Remove the embeddings, NN, re-count the expression data
adata.obsm.clear()
adata.obsp.clear()
adata.X = adata.layers['counts'].copy()
adata.varm.clear()
var_cols_to_clear = ['n_cells', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'mean', 'std']
adata.var.drop(columns=var_cols_to_clear, inplace=True)
adata.uns.clear()
# Remove the log1p
adata.layers.clear()
# Keep counts
adata.layers['counts'] = adata.X.copy()

# Save into the results dir
adata.write_h5ad(f"results/{tissue}/objects/adata_manual_lineage_clean.h5ad")

# Dvide by manual lineage and save
lins = np.unique(adata.obs['manual_lineage'])
for l in lins:
    print(l)
    temp = adata[adata.obs['manual_lineage'] == l]
    print(temp.shape)
    temp.write_h5ad(f"input_cluster_within_lineage/adata_manual_lineage_clean_{tissue}_{l}.h5ad")