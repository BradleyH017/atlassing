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

# Add provisional category annotation and colours
annots = pd.read_csv("temp/2024-08-28_provisional_annot_cats.csv")
colours = pd.read_csv("temp/master_colours.csv")

# Bind
adata.obs.reset_index(inplace=True)
adata.obs = adata.obs.merge(annots, on="leiden", how="left")
adata.obs.set_index("cell", inplace=True)

# Define map
color_map = dict(zip(colours['Category'], colours['Colour code']))


# Make overall plots:
overall_plot = ["tissue", "Category", "Label_1", "disease_status"]
adata.obs['tissue'].replace('r', 'Rectum', inplace=True)
adata.obs['tissue'].replace('ti', 'TI', inplace=True)
adata.obs['tissue'].replace('blood', 'Blood', inplace=True)
for c in overall_plot:
    print(c)
    if c in ["Category", "tissue"]:
        sc.pl.umap(adata, color=c, legend_loc="on data", frameon=False, palette=color_map, title=None, legend_fontoutline=1, legend_fontsize=10, save=f'_overall_{c}.png', show=False)
        sc.pl.umap(adata, color=c, frameon=False, palette=color_map, title=None, legend_fontsize=10, save=f'_overall_{c}_legend_off.png', show=False)
    if c == "Label_1":
        sc.pl.umap(adata, color=c, frameon=False, cmap='Set1', title=None, legend_fontsize=10, save=f'_overall_{c}.png', show=False)
    else: 
        sc.pl.umap(adata, color=c, frameon=False, title=None, legend_fontsize=10, save=f'_overall_{c}.png', show=False)

# Plot within category
cats = np.unique(adata.obs['Category'])
#adata.obs = adata.obs.strings_to_categoricals()
#adata.obs['Label_1'] = adata.obs['Label_1'].astype('category')
#adata.obs['Label_1'] = adata.obs['Label_1'].astype('category')
for c in cats:
    print(c)
    #adata_subset = adata[adata.obs['Category'] == c].copy() 
    sc.pl.umap(adata[adata.obs['Category'] == c], color="Label_1", cmap='Set1', frameon=False, title=None, legend_fontsize=10, save=f'_within_Category_{c}.png', show=False)


# Plot for some specific categories / cell-states (grey out the rest)
level = ["tissue", "Category", "Label_1", "Category", "Label_1"]
annot = ["Rectum", "Colonocyte", "Colonocyte (1)", "T", "CD8 tissue resident memory (1)"]
tissue = ["Rectum", "Rectum", "Rectum", "TI", "TI"]
cols = ["#E07B28", "#E07B28", "#E07B28", "#D640A1", "#D640A1"]
for i, l in enumerate(level):
    adata.obs['plot'] = (adata.obs[level[i]] == annot[i]) & (adata.obs.tissue == tissue[i])
    adata.obs['plot'] = adata.obs['plot'].replace({False: "No", True: "Yes"})
    fname = f"{tissue[i]}-{level[i]}-{annot[i]}"
    temp_cols_map = dict({"Yes": cols[i], "No": "#D3D3D3"})
    sc.pl.umap(adata, color="plot", frameon=False, palette = temp_cols_map, title=None, legend_fontsize=10, save=f'_{fname}.png', show=False)
