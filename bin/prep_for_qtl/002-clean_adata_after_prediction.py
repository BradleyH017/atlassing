#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-08-7'
__version__ = '0.0.1'

import scanpy as sc
import pandas as pd 
import numpy as np


# Define adata (complete, annotated), columns to threshold and filter values
h5 = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/results/combined/objects/celltypist_prediction.h5ad"
filters = dict({"conf_score": 0.5, "pct_counts_gene_group__mito_transcript": 50, "log_n_genes_by_counts": np.log10(250), "log_total_counts": np.log10(500)})
over_under = {"conf_score": "over", "pct_counts_gene_group__mito_transcript": "under", "log_n_genes_by_counts": "over", "log_total_counts": "over"}

# Load in the anndata object
adata = sc.read_h5ad(h5)
adata.obs['log_n_genes_by_counts'] = np.log10(adata.obs['n_genes_by_counts'])
adata.obs['log_total_counts'] = np.log10(adata.obs['total_counts'])

# Save a list of the genotypes we have
genos = np.unique(adata.obs['Genotyping_ID'].astype(str))
genos = genos[genos != 'nan']
with open(f"/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/results/combined/tables/genos_from_expression.txt", 'w') as file:
    for g in genos:
        file.write(g + '\n')

# Filter 
print(f"Original adata shape: {adata.shape}")
for col, value in filters.items():
    print(f"col: {col}, value: {value}, over_under: {over_under[col]}")
    if over_under[col] == "under":
        adata = adata[adata.obs[col] < value]
    if over_under[col] == "over":
        adata = adata[adata.obs[col] > value]
    #
    print(f"After filter for {col}, adata shape: {adata.shape}")

# Merge with the categories
annots = pd.read_csv("temp/2024-08-28_provisional_annot_cats.csv")

# Bind
annots.rename(columns={"leiden": "predicted_labels", "Category": "predicted_category"}, inplace=True)
adata.obs.reset_index(inplace=True)
adata.obs = adata.obs.merge(annots, on="predicted_labels", how="left")
adata.obs.set_index("cell", inplace=True)

# Add unnatotated
adata.obs['unannotated'] = "unannotated"

# Bind each of these to tissue and save to a new column
for c in ["predicted_labels", "predicted_category", "unannotated"]:
    print(c)
    adata.obs[f"{c}_tissue"] = adata.obs[c].astype(str) + "_" + adata.obs['tissue'].astype(str)

# First step actually uses the log1p_cp10k expression matrix, but under a different name. So rename this and save lots of mem
adata.layers['cp10k'] = adata.layers.pop('log1p_cp10k')

# Make sure .var index is "ENS" (replaced during prediction)
adata.var.set_index("ENS", inplace=True)

# Save
adata.write_h5ad("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/results/combined/objects/celltypist_0.5_ngene_ncount_mt_filt.h5ad")

# Exclude samples with 'Missing' genotypes. As there is > 1 sample with this annotation they may skew the PCA calculation after psuedobulking
sample_geno = adata.obs[["samp_tissue", "Genotyping_ID"]].reset_index(drop=True).drop_duplicates()
nomiss = adata[adata.obs['Genotyping_ID'] != "Missing"]
nomiss = nomiss[nomiss.obs['Genotyping_ID'].astype(str) != "nan"]
nomiss.write_h5ad("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/results/combined/objects/celltypist_0.5_ngene_ncount_mt_filt_nomiss.h5ad")


# Save a version downsampled to 5k cells for each cluster (best conf score)
predicted_labels = adata.obs['predicted_labels']
conf_score = adata.obs['conf_score']
selected_indices = []
for label in predicted_labels.unique():
    print(label)
    label_indices = adata.obs[predicted_labels == label].index
    label_cells = adata[label_indices]
    if label_cells.shape[0] > 5000:
        print(">5k cells - selecting")
        top_cells = label_cells.obs.sort_values(by='conf_score', ascending=False).head(5000).index
        selected_indices.extend(top_cells)
    else:
        selected_indices.extend(label_indices)

ds = adata[selected_indices].copy()
ds.shape

# Save this
ds.write_h5ad("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/results/combined/objects/celltypist_0.5_ngene_ncount_mt_filt_downsampled.h5ad")


# Divide by tissue
tissues = np.unique(adata.obs['tissue'])
for t in tissues:
    print(t)
    adata[adata.obs['tissue'] == t].write_h5ad(f"/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/results/combined/objects/celltypist_0.5_ngene_ncount_mt_filt_{t}.h5ad")