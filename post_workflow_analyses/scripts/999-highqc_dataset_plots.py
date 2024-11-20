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
    # Also plot with the correct colour for this subset
    sc.pl.umap(adata, color=c, frameon=False, palette=color_map, title=None, legend_fontsize=10, save=f'_overall_{c}_legend_off.png', show=False)

# Plot for some specific categories / cell-states (grey out the rest)
level = ["tissue", "Category", "Label_1", "Category", "Label_1", "tissue", "Label_1", "Category", "Category", "Category", "Label_1"]
annot = ["Rectum", "Colonocyte", "Colonocyte (1)", "T", "CD8 tissue resident memory (1)", "TI", "Intermediate monocyte SOD2+", "Myeloid", "Secretory", "Stem", "Rectal Goblet (1)"]
tissue = ["Rectum", "Rectum", "Rectum", "TI", "TI", "TI", "all", "all", "Rectum", "Rectum", "Rectum"]
cols = ["#E07B28", "#E07B28", "#E07B28", "#D640A1", "#D640A1", "#117733", "#AA4499", "#AA4499", "#332288", "#4DCFFA", "#332288"]
for i, l in enumerate(level):
    if tissue[i] == "all":
        adata.obs['plot'] = (adata.obs[level[i]] == annot[i])
    else:
        adata.obs['plot'] = (adata.obs[level[i]] == annot[i]) & (adata.obs.tissue == tissue[i])
    #
    adata.obs['plot'] = adata.obs['plot'].replace({False: "No", True: "Yes"})
    fname = f"{tissue[i]}-{level[i]}-{annot[i]}"
    temp_cols_map = dict({"Yes": cols[i], "No": "#D3D3D3"})
    sc.pl.umap(adata, color="plot", frameon=False, palette = temp_cols_map, title=None, legend_fontsize=10, save=f'_{fname}.png', show=False)


# Plot the proportion of categories represented across samples from each tissue
proportion_df = adata.obs.groupby(['samp_tissue', 'Category']).size().reset_index(name='count')
proportion_df['proportion'] = proportion_df.groupby('samp_tissue')['count'].transform(lambda x: x / x.sum())
pivot_df = proportion_df.pivot(index='samp_tissue', columns='Category', values='proportion').fillna(0)
pivot_df = pivot_df.reset_index()
pivot_df['Tissue'] = pivot_df['samp_tissue'].str.split('_').str[-1]
tissue_groups = pivot_df.groupby('Tissue')
from scipy.cluster.hierarchy import linkage, leaves_list
for tissue, group in tissue_groups:
    # Set samp_tissue as the index for hierarchical clustering
    group = group.set_index('samp_tissue')
    proportions = group[group.columns[:-1]]  # Exclude Tissue column
    linkage_matrix = linkage(proportions, method='average')
    clustered_order = leaves_list(linkage_matrix)
    group = group.iloc[clustered_order]
    x_positions = np.arange(len(group))
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(group))
    bar_width = 1.0  # Ensure no gaps between bars
    for idx, category in enumerate(group.columns[:-1]):  # Exclude Tissue
        ax.bar(
            x_positions, 
            group[category], 
            bottom=bottom, 
            label=category, 
            color=color_map.get(category, '#000000'),  # Default to black if not in color_map
            width=bar_width
        )
        bottom += group[category].values
    #
    ax.set_xlim(-0.5, len(x_positions) - 0.5)  # Ensures bars span the full width
    ax.grid(False)
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f'Relative Proportions of Cell Categories - {tissue}')
    ax.set_ylabel('Proportion')
    ax.set_xticks([])
    # Save the figure
    plt.savefig(f"results_round3/combined/figures/category_proportions_{tissue}.png", bbox_inches='tight')
    plt.clf()



############# SCRAP
# Explore metadata
obs = adata.obs.copy()
ncells_per_sample = 10
min_perc_single_tissue = 0.25
minsamples = 40

# For each cluster, calculate the number of samples with > x cells and the overall proportion of cells coming from each tissue
# Percentage single tissue
perc_tissue = obs.groupby("leiden")["tissue"].value_counts(normalize=True).reset_index()
perc_tissue['leiden_tissue'] = perc_tissue['leiden'].astype(str) + "_" + perc_tissue['tissue'].astype(str)

# NSamples with gt nCells per sample
results = []
for (leiden, tissue), group in obs.groupby(['leiden', 'tissue']):
    samp_tissue_counts = group['samp_tissue'].value_counts()
    count_meeting_criteria = (samp_tissue_counts >= ncells_per_sample).sum()
    # Append the results
    results.append({
        'leiden': leiden,
        'tissue': tissue,
        'nsamp_gt_min_ncells': count_meeting_criteria
    })

nsamples = pd.DataFrame(results)
nsamples['leiden_tissue'] = nsamples['leiden'].astype(str) + "_" + nsamples['tissue'].astype(str)

# Combine
sumstats = nsamples.merge(perc_tissue[["leiden_tissue", "proportion"]], how="left", on="leiden_tissue")
sumstats['testable'] = (sumstats['proportion'] > min_perc_single_tissue) & (sumstats['nsamp_gt_min_ncells'] > minsamples)

# Summarise the testable comparisons
testmeta = []
for leiden, group in sumstats.groupby('leiden'):
    testable = group[group['testable']]
    ntissues = testable.shape[0]
    tissues = ",".join(testable['tissue'])
    testmeta.append({
        'leiden': leiden,
        'ntissues': ntissues,
        'testable_tissues': tissues
    })

testmeta = pd.DataFrame(testmeta)
testmeta.to_csv("results_round3/combined/tables/DE_test_clusters.csv", index = False)