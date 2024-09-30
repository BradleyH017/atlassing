#!/usr/bin/env python
# single_cell env! 

__author__ = 'Bradley Harris'
__date__ = '2024-01-12'
__version__ = '0.0.1'

# Change dir
import os
cwd = os.getcwd()
print(cwd)

# Load packages
import sys
sys.path.append('/software/team152/bh18/pip')
sys.path.append('/usr/local/')
print("System path")
print(sys.path)
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
print("Numpy file")
import scib_metrics 
from pytorch_lightning import Trainer
from scib_metrics.utils import principal_component_regression
from anndata.experimental import read_elem
from h5py import File
print("Loaded libraries")

# load in the final high QC object post-integration
f = "results_round3/combined/objects/adata_PCAd_batched.h5ad"
f2 = File(f, 'r')
obs = read_elem(f2['obs'])

# Merge categories
annots = pd.read_csv("temp/2024-08-28_provisional_annot_cats.csv")
obs.reset_index(inplace=True)
obs = obs.merge(annots, on="leiden", how="left")
obs.set_index("cell", inplace=True)
lins = np.unique(obs['manual_lineage'])
lins = np.array(np.append(lins, "all")) # add all
chuck = np.array(["Mast", "Platelet+RBC"])
lins=lins[~np.isin(lins, chuck)]
#del obs

# Define cols to plot
var_explained_cols=["log_n_genes_by_counts", "log_total_counts", "pct_counts_gene_group__mito_transcript", "patient_id", "tissue", "sex", "age", "disease_status", "inflammation_status", "samp_tissue"]

# For each lineage, load in the data and compute variance explained for the given covariates
within_lineage_dir="results_round2"
across_all_f="results_round3/combined/objects/adata_PCAd_batched.h5ad"
var_df = pd.DataFrame(index=var_explained_cols, columns=lins)
for l in lins:
    print(f"~~~~~ Working on: {l}")
    if l != "all":
        input=f"{within_lineage_dir}/all_{l}/objects/adata_PCAd_batched.h5ad"
        adata = sc.read_h5ad(input)
    else:
        adata = sc.read_h5ad(across_all_f)
    var_explained_cols = np.intersect1d(var_explained_cols, adata.obs.columns)
    print("Computing variance explained")
    for v in var_explained_cols:
        if v in adata.obs.columns:
            print(f"{v}")
            if pd.api.types.is_numeric_dtype(adata.obs[v]):
                var_df.loc[v, l] = scib_metrics.utils.principal_component_regression(adata.obsm["X_scVI"], adata.obs[v], categorical=False)
            else:
                var_df.loc[v, l] = scib_metrics.utils.principal_component_regression(adata.obsm["X_scVI"], adata.obs[v], categorical=True)
    #
    print(var_df)
    
var_df.to_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/results/combined/tables/variance_explained_per_lineage_and_all.csv")


