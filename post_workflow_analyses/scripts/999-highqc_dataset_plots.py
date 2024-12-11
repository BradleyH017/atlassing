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


# Make the round 1 lineage annotation plot
sc.settings.figdir="other_paper_figs"
sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")

# Load the anndata object from round 1
r1 = sc.read_h5ad("results_round1/all/tables/clustering_array/leiden_0.03/adata_PCAd_batched_umap_0.03.h5ad")
umap = r1.obsm['UMAP_X_scVI']
clusters = r1.obs['leiden']
del r1
adata = sc.read_h5ad("input_v7/adata_raw_input_all.h5ad")
adata = adata[adata.obs.index.isin(clusters.index)]
adata.obsm['X_umap'] = umap
adata.obs['leiden'] = clusters

major = {"Epithelial": ["EPCAM" ,'CDH1', 'KRT19'], "Mesenchymal": ["COL1A1","COL1A2","COL6A2","VWF"], 'B':['CD79A', 'MS4A1', 'MS4A1', 'CD79B'], 'Plasma':['MZB1', 'JCHAIN'], 'T_NK':['CD3D', 'CD3E', 'CD3G','CCR7','IL7R', 'TRAC'], 'Myeloid':['ITGAM', 'CD14', 'CSF1R', 'TYROBP'], 'Mast':["TPSAB1", 'TPSB2', "CPA3" ], 'Platelet+RBC':['GATA1', 'TAL1', 'ITGA2B', 'ITGB3', 'GP1BA', 'PPBP', 'PF4', 'TUBB1', 'THBS1']}
sc.pl.dotplot(adata, major, layer="log1p_cp10k", gene_symbols = "gene_symbols", groupby='leiden', dendrogram=False, save="_r1_major_lineage_markers.png")

# Generate maps of clusters --> manual lineages
manual_mapping = {"0": "Epithelial", "1": "T-NK", "2": "B", "3": "Myeloid", "4": "B", "5": "Mesenchymal", "6": "Epithelial", "7": "Mast", "8": "Platelet"}
adata.obs['manual_lineage'] = adata.obs['leiden'].map(manual_mapping)

# Derive colours
colours = pd.read_csv("temp/master_colours.csv")
colours['manual_lineage'] = colours['Category'] # Plotting the manual lineages here
color_map = dict(zip(colours['manual_lineage'], colours['Colour code']))

# Plot UMAPs
sc.pl.umap(adata, color='manual_lineage', legend_loc="on data", frameon=False, palette=color_map, title=None, legend_fontoutline=1, legend_fontsize=10, save=f'_overall_r1.png', show=False)
sc.pl.umap(adata, color='manual_lineage', frameon=False, palette=color_map, title=None, legend_fontsize=10, save=f'_overall_r1_legend_off.png', show=False)
sc.pl.umap(adata, color='leiden', legend_loc="on data", frameon=False, palette='Set2', title=None, legend_fontoutline=1, legend_fontsize=10, save=f'_overall_r1_leiden.png', show=False)
sc.pl.umap(adata, color='leiden', frameon=False, palette='Set2', title=None, legend_fontsize=10, save=f'_overall_r1_leiden_legend_off.png', show=False)


############ Plot results from WITHIN major lineages
del adata
lins_res = {"all_Epithelial": 0.9, "all_Mesenchymal": 1.0, "all_Myeloid": 0.7, "all_B": 0.8, "all_T": 1.1, "Platelet+RBC": np.nan, "Mast": np.nan}
chuck = {"all_Epithelial": ["11", "33", "34"], "all_Mesenchymal": ["15"], "all_Myeloid": ["6", "13", "18"], "all_B": ["14", "15"], "all_T": ["17", "18", "23", "25"]}
additional_chuck = {"all_Epithelial": ["8"], "all_Mesenchymal": ["10"], "all_B": ["7"], "all_Myeloid": ["12", "5", "15"]}
chuck_additional = True

linadata = []
final_clusters = []
for l in lins_res:
    print(f"Working on: {l}")
    res = lins_res[l]
    if not np.isnan(res):
        adata = sc.read_h5ad(f"results_round2/{l}/tables/clustering_array/leiden_{str(res)}/adata_PCAd_batched_umap_{str(res)}.h5ad")
        clusters = adata.obs[['leiden']].reset_index()
        clusters = clusters[~clusters['leiden'].astype(str).isin(chuck[l])]
        if chuck_additional and l in additional_chuck.keys():
            clusters = clusters[~clusters['leiden'].astype(str).isin(additional_chuck[l])]
        pref = l.replace("all_", "")
        clusters['leiden'] = pref + "_" + clusters['leiden'].astype(str)
    else:
        clusters = pd.DataFrame({"cell": adata.obs[adata.obs['manual_lineage'] == l].index.values, "leiden": l})
    
    adata.obs['pref_leiden'] = pref + "_" + adata.obs['leiden'].astype(str)
    adata = adata[adata.obs.index.isin(clusters['cell'])]
    linadata.append(adata)
    final_clusters.append(clusters)

final_clusters = pd.concat(final_clusters)

# Plot
for index, l in enumerate(lins_res.keys()):
    print(f"~~~ Plotting {l} ~~~")
    sc.pl.umap(linadata[index], color='leiden', legend_loc="on data", frameon=False, title=None, legend_fontoutline=1, legend_fontsize=10, save=f'_{l}_good_clusters.png', show=False)
    sc.pl.umap(linadata[index], color='leiden', frameon=False, title=None, legend_fontoutline=1, legend_fontsize=10, save=f'_{l}_good_clusters_legend_off.png', show=False)



############ Plot the results from high quality data ACROSS major lineages
# Load anndata (backed)
adata = sc.read('/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/results_round3/combined/objects/adata_PCAd_batched_umap.h5ad', backed='r')

# Add provisional category annotation and colours
annots = pd.read_csv("temp/2024-12-02_annot_cats.csv") # post annotation jamboree
colours = pd.read_csv("temp/master_colours.csv")

# Bind
adata.obs.reset_index(inplace=True)
adata.obs = adata.obs.merge(annots, on="leiden", how="left")
adata.obs.set_index("cell", inplace=True)

# Define map
color_map = dict(zip(colours['Category'], colours['Colour code']))

# Make overall plots:
overall_plot = ["tissue", "Category", "JAMBOREE_ANNOTATION", "disease_status"]
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
    sc.pl.umap(adata[adata.obs['Category'] == c], color="JAMBOREE_ANNOTATION", cmap='Set1', frameon=False, title=None, legend_fontsize=10, save=f'_within_Category_{c}.png', show=False)
    # Also plot with the correct colour for this subset
    #sc.pl.umap(adata, color=c, frameon=False, palette=color_map, title=None, legend_fontsize=10, save=f'_overall_{c}_legend_off.png', show=False)

# Plot within tissue also
tissues = np.unique(adata.obs['tissue'])
for t in tissues:
    print(c)
    sc.pl.umap(adata[adata.obs['tissue'] == t], color="JAMBOREE_ANNOTATION", cmap='Set1', frameon=False, title=None, legend_fontsize=10, save=f'_label_within_{t}.png', show=False)
    sc.pl.umap(adata[adata.obs['tissue'] == t], color="tissue", cmap='Set1', frameon=False, title=None, legend_fontsize=10, save=f'_tissue_within_{t}.png', show=False)
    sc.pl.umap(adata[adata.obs['tissue'] == t], color="Category", cmap='Set1', frameon=False, title=None, legend_fontsize=10, save=f'_Category_within_{t}.png', show=False)


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

# Plot the number of cells from each tissue contributing to each cluster
ml = np.unique(adata.obs['manual_lineage'])
col="leiden"
for l in ml:
    print(f"---- Working on {l} -----")
    proportion_df = adata.obs[adata.obs['manual_lineage'] == l].groupby([col, 'tissue']).size().reset_index(name='count')
    proportion_df['proportion'] = proportion_df.groupby(col)['count'].transform(lambda x: x / x.sum())
    pivot_df = proportion_df.pivot(index=col, columns="tissue", values='proportion').fillna(0)
    pivot_df = pivot_df[pivot_df.sum(axis=1) > 0]
    color_mapping = {"r": "#E07B28", "blood": "#BB5566", "ti": "#117733"}
    fig, ax = plt.subplots(figsize=(15, 8))
    bottom = np.zeros(len(pivot_df))
    #
    for idx, category in enumerate(pivot_df.columns):
        ax.bar(pivot_df.index, pivot_df[category], bottom=bottom, color=color_mapping.get(category, "#000000"), label=category)
        bottom += pivot_df[category].fillna(0).values
    #
    ax.set_xticklabels(pivot_df.index, rotation=45, ha='right', rotation_mode='anchor')
    ax.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('Relative Proportions of Tissue by Cluster (cell number)')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Proportion')
    plt.savefig(f"results_round3/combined/figures/tissue__across_clusters_{l}.png", bbox_inches='tight')
    plt.clf()

# Rather, save this as a table. 
# Cells
proportion_df = adata.obs.groupby([col, 'tissue']).size().reset_index(name='cell_count')
proportion_df['cell_proportion'] = proportion_df.groupby(col)['cell_count'].transform(lambda x: x / x.sum())
pivot_df = proportion_df.pivot(index=col, columns="tissue", values='cell_proportion').fillna(0)
pivot_df.columns = pivot_df.columns.astype(str) + "_cell_prop"
# Samples
grouped = adata.obs.groupby([col, 'tissue', 'samp_tissue']).size().reset_index(name='sample_count')
filtered_group = grouped[grouped['sample_count'] > 5]
proportion_df_sample = filtered_group.groupby([col, 'tissue']).size().reset_index(name='sample_count')
proportion_df_sample['sample_proportion'] = proportion_df_sample.groupby(col)['sample_count'].transform(lambda x: x / x.sum())
pivot_df_sample = proportion_df_sample.pivot(index=col, columns="tissue", values='sample_proportion').fillna(0)
pivot_df_sample.columns = pivot_df_sample.columns.astype(str) + "_sample_prop"
merged_df = pivot_df_sample.join(pivot_df, on="leiden", how="inner")
merged_df.to_csv("results_round3/combined/tables/cell_sample_prop_per_leiden.csv")

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