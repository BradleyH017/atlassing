#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-08-12'
__version__ = '0.0.1'

# Load in the libraries
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from anndata.experimental import read_elem
from h5py import File
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('bin')
import nmad_qc # Import custom functions

# Import obs from the round 1 results
f = f"results_round1/all/objects/adata_manual_lineage_clean.h5ad"
f2 = File(f, 'r')
# Read only cell-metadata
obs = read_elem(f2['obs'])

# Define pathout
outdir="../results/explore_round1"
if os.path.exists(outdir) == False:
        os.mkdir(outdir)
        
# Define cluster col / relative nmad
col="leiden_0.025_round1_QC"
relative_nMAD_threshold=3

# Make dummy anndata
adata = ad.AnnData(obs=obs)

# For each cluster summarise the proportion of cells coming from each tissue x disease status
proportion_df = adata.obs.groupby([col, 'tissue_disease']).size().reset_index(name='count')
proportion_df['proportion'] = proportion_df.groupby(col)['count'].transform(lambda x: x / x.sum())
pivot_df = proportion_df.pivot(index=col, columns="tissue_disease", values='proportion').fillna(0)
colors = sns.color_palette("husl", len(pivot_df.columns))
fig, ax = plt.subplots(figsize=(10, 6))
bottom = np.zeros(len(pivot_df))

for idx, category in enumerate(pivot_df.columns):
    ax.bar(pivot_df.index, pivot_df[category], bottom=bottom, color=colors[idx], label=category)
    bottom += pivot_df[category].fillna(0).values

ax.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_title('Relative Proportions of Tissue x Disease by Cluster')
ax.set_xlabel('Cluster')
ax.set_ylabel('Proportion')
plt.savefig(f"{outdir}/tissue_disease_across_clusters.png", bbox_inches='tight')
plt.clf()

# Plot the QC per cell per cluster
cols = ["pct_counts_gene_group__mito_transcript", "log_n_genes_by_counts", "log_total_counts"]
thresholds = {"pct_counts_gene_group__mito_transcript": {"blood": 20, "r": 50, "ti": 50}, "log_n_genes_by_counts": np.log10(250), "log_total_counts": np.log10(500)}
over_under = {"pct_counts_gene_group__mito_transcript": "under", "log_n_genes_by_counts": "over", "log_total_counts": "over"}
min_ncells_per_sample=10
adata.obs['log_n_genes_by_counts'] = np.log10(adata.obs['n_genes_by_counts'])
adata.obs['log_total_counts'] = np.log10(adata.obs['total_counts'])
clusters = np.unique(adata.obs[col])
for k in clusters:
    print(f"~~~~ Cluster: {k} ~~~~~")
    loutdir=f"{outdir}/cluster_{k}"
    if os.path.exists(loutdir) == False:
        os.mkdir(loutdir)
    # Subset
    temp = adata[adata.obs[col] == k]
    # Chuck samps with few cells
    samp_data = np.unique(temp.obs.samp_tissue, return_counts=True)
    cells_sample = pd.DataFrame({'sample': samp_data[0], 'Ncells':samp_data[1]})
    chuck = cells_sample.loc[cells_sample['Ncells'] < min_ncells_per_sample, 'sample'].values
    if len(chuck) > 0:
        temp = temp[~temp.obs['samp_tissue'].isin(chuck)]
    # Plot
    for c in cols:
        print(f"~~~~ {c} ~~~~~")
        nmad_qc.dist_plot(temp, c, within="tissue", relative_threshold=relative_nMAD_threshold, absolute=thresholds[c], out=loutdir)


# test the relative threshold function
c="pct_counts_gene_group__mito_transcript"
nmad_qc.update_obs_qc_plot_thresh(adata, c, within="tissue", relative_threshold=10, threshold_method = "None", relative_directionality = "bi", absolute=thresholds[c], absolute_directionality = over_under[c], plot =True, out=outdir, out_suffix = "_THRESHOLDED")
