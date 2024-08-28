# Bradley August 2024
# Plotting demographics of the IBDverse cohort

# Load libraries
import scanpy as sc
import pandas as pd
import numpy as np
import os
from anndata.experimental import read_elem
from h5py import File
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotnine as plt9
from matplotlib.lines import Line2D

# Define outdir
sc.settings.figdir="results_round3/combined/figures/UMAP"
sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")

# Load anndata (backed)
adata = sc.read('/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/results_round3/combined/objects/adata_PCAd_batched_umap.h5ad', backed='r')

# Add provisional category annotation
annots = pd.read_csv("temp/2024-08-28_provisional_annot_cats.csv")

# Bind
adata.obs.reset_index(inplace=True)
adata.obs = adata.obs.merge(annots, on="leiden", how="left")

# Make overall plots:
overall_plot = ["tissue", "Category", "Label_1", "disease_status"]
for c in overall_plot:
    print(c)
    if c == "Category":
        sc.pl.umap(adata, color=c, legend_loc="on data", frameon=False, title=None, legend_fontoutline=1, legend_fontsize=10, save=f'_overall_{c}.png', show=False)
    else: 
        sc.pl.umap(adata, color=c, frameon=False, title=None, legend_fontsize=10, save=f'_overall_{c}.png', show=False)

# Plot within category
cats = np.unique(adata.obs['Category'])
for c in cats:
    print(c)
    sc.pl.umap(adata[adata.obs['Category'] == c], color="Label_1", frameon=False, title=None, legend_fontsize=10, save=f'_within_Category_{c}.png', show=False)
