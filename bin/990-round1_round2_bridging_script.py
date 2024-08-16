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
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('bin')
import nmad_qc # Import custom functions
import anndata as ad


# Define tissue and resolution to define lineages from (selection of clusters for each lineage will vary depending on this)
tissue="all"
resolution="0.03"

# Import data
fpath = f"results/{tissue}/tables/clustering_array/leiden_{resolution}/adata_PCAd_batched_umap_{resolution}.h5ad"
old = sc.read_h5ad(fpath)

# Plot with labels on data for ease
sc.settings.figdir=f"results/{tissue}/figures/UMAP/annotation"
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")
sc.pl.umap(old, color="leiden", legend_loc="on data", save=f"_{tissue}_leiden_{resolution}_legend_on.png")

# Generate additional UMAPs from non HVG expression object
umap = old.obsm['UMAP_X_scVI']
clusters = old.obs['leiden']
del old
adata = sc.read_h5ad("input_v7/adata_raw_input_all.h5ad")
adata = adata[adata.obs.index.isin(clusters.index)]
adata.obsm['X_umap'] = umap
# Plot missing
missing="EPCAM,KRT8,KRT18,CDH5,COL1A1,COL1A2,COL6A2,VWF,PTPRC,CD3D,CD3G,CD3E,CD79A,CD79B,CD14,FCGR3A,CD68,CD83,CSF1R,FCER1G"
missing = missing.split(",")
exprfigpath = f"results/{tissue}/figures/UMAP/expr"
sc.settings.figdir=exprfigpath
adata.layers['counts'] = adata.X.copy()
adata.X = adata.layers['log1p_cp10k'].copy()
for c in missing:
    ens=adata.var[adata.var['gene_symbols'] == c].index[0]
    sc.pl.umap(adata, layer="log1p_cp10k", color = ens, save="_scVI_" + c + ".png")

# Also plot a dotplot
adata.obs['leiden'] = clusters
major = {"Epithelial": ["EPCAM" ,'CDH1', 'KRT19', 'EPCAM'], "Mesenchymal": ["COL1A1","COL1A2","COL6A2","VWF"], 'Immune':['PTPRC'], 'B':['CD79A', 'MS4A1', 'MS4A1', 'CD79B'], 'Plasma':['MZB1', 'JCHAIN'], 'T':['CD3D', 'CD3E', 'CD3G','CCR7','IL7R', 'TRAC'], 'Myeloid':['ITGAM', 'CD14', 'CSF1R', 'TYROBP'], 'DC':['ITGAX', 'CLEC4C','CD1C', 'FCER1A', 'CLEC10A'], 'Mac':['APOE', 'C1QA', 'CD68','AIF1'], 'Mono':['FCN1','S100A8', 'S100A9', "CD14", "FCGR3A", 'LYZ'], 'Mast':["TPSAB1", 'TPSB2', "CPA3" ], 'Platelet+RBC':['GATA1', 'TAL1', 'ITGA2B', 'ITGB3']}
sc.pl.dotplot(adata, major, layer="log1p_cp10k", gene_symbols = "gene_symbols", groupby='leiden', dendrogram=False, save="_scVI_major_markers.png")

# Summarise the top 2 annotations per cluster for a range of annotation columns
annot_cols = ["Keras:predicted_celltype", "Azimuth:predicted.celltype.l1", "Azimuth:predicted.celltype.l2", "Celltypist:megagut_celltypist_lowerGI+lym_adult_mar24:predicted_labels"]
def most_and_second_most_frequent(series):
    counts = series.value_counts()
    most_common = counts.index[0]
    most_common_pct = counts.iloc[0] / counts.sum() * 100
    if len(counts) > 1:
        second_most_common = counts.index[1]
        second_most_common_pct = counts.iloc[1] / counts.sum() * 100
        return f"{most_common} ({most_common_pct:.1f}%), {second_most_common} ({second_most_common_pct:.1f}%)"
    else:
        return f"{most_common} ({most_common_pct:.1f}%)"

# Group by 'leiden' and apply the custom function to each annot_col
result = adata.obs.groupby('leiden')[annot_cols].agg(lambda x: most_and_second_most_frequent(x))
result.to_csv(f"results/{tissue}/tables/clustering_array/leiden_{resolution}/top_votes_other_annotations.csv")

# Look at counts
counts = pd.DataFrame(adata.obs['leiden'].value_counts())
counts.to_csv(f"results/{tissue}/tables/clustering_array/leiden_{resolution}/cluster_counts.csv")


# Additional plots: look at the proportion of tissue_disease values in each cluster
col="leiden"
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
plt.savefig(f"results/{tissue}/figures/UMAP/annotation/tissue_disease_across_clusters.png", bbox_inches='tight')
plt.clf()

# Have a look at QC similarity
cols = ["pct_counts_gene_group__mito_transcript", "log_n_genes_by_counts", "log_total_counts"]
thresholds = {"pct_counts_gene_group__mito_transcript": {"blood": 20, "r": 50, "ti": 50}, "log_n_genes_by_counts": np.log10(250), "log_total_counts": np.log10(500)}
over_under = {"pct_counts_gene_group__mito_transcript": "under", "log_n_genes_by_counts": "over", "log_total_counts": "over"}
min_ncells_per_sample=10
adata.obs['log_n_genes_by_counts'] = np.log10(adata.obs['n_genes_by_counts'])
adata.obs['log_total_counts'] = np.log10(adata.obs['total_counts'])
clusters = np.unique(adata.obs[col])
check_qc_dir = f"results/{tissue}/figures/UMAP/annotation"
epithelial_clusters = ["0", "6"]
relative_nMAD_threshold=3
for k in clusters:
    print(f"~~~~ Cluster: {k} ~~~~~")
    loutdir=f"{check_qc_dir}/cluster_{k}"
    if os.path.exists(loutdir) == False:
        os.mkdir(loutdir)
    # Subset
    temp = adata[adata.obs[col] == k]
    if k in epithelial_clusters:
        print("Removing blood")
        temp = temp[temp.obs['tissue'] != "blood"]
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

deep_dive = T
if deep_dive:
    # If we group cluster 2 and 4? 
    bgroup = adata[adata.obs['leiden'].isin(["2", "4"])]
    k="2_and_4"
    loutdir=f"{check_qc_dir}/cluster_{k}"
    if os.path.exists(loutdir) == False:
        os.mkdir(loutdir)

    for c in cols:
        print(f"~~~~ {c} ~~~~~")
        nmad_qc.dist_plot(bgroup, c, within="tissue", relative_threshold=relative_nMAD_threshold, absolute=thresholds[c], out=loutdir)

    # Compare the number of cells from each tissue being kept/lost with this threshold for each 
    check_clusters = ["2", "4"]
    sum = []
    tempobs = []
    for k in check_clusters:
        loutdir=f"{check_qc_dir}/cluster_{k}"
        temp = adata[adata.obs['leiden'] == k]
        for c in cols:
            nmad_qc.update_obs_qc_plot_thresh(temp, c, within="tissue", relative_threshold=relative_nMAD_threshold, threshold_method = "outer", relative_directionality = "bi", absolute=thresholds[c], absolute_directionality = over_under[c], plot =True, out=loutdir, out_suffix = "_THRESHOLDED")
        #
        keep_columns = temp.obs.filter(like='_keep').columns
        temp.obs['cell_keep'] = temp.obs[keep_columns].all(axis=1)
        summary = temp.obs.groupby('tissue')['cell_keep'].value_counts(normalize=True).unstack(fill_value=0)
        sum.append(summary)
        tempobs.append(temp[temp.obs['cell_keep']])

    # Same but if combined
    bgroup = adata[adata.obs['leiden'].isin(["2", "4"])]
    k="2_and_4"
    loutdir=f"{check_qc_dir}/cluster_{k}"
    for c in cols:
        nmad_qc.update_obs_qc_plot_thresh(bgroup, c, within="tissue", relative_threshold=relative_nMAD_threshold, threshold_method = "outer", relative_directionality = "bi", absolute=thresholds[c], absolute_directionality = over_under[c], plot =True, out=loutdir, out_suffix = "_THRESHOLDED")

    keep_columns = bgroup.obs.filter(like='_keep').columns
    bgroup.obs['cell_keep'] = bgroup.obs[keep_columns].all(axis=1)
    bgroup_keep = bgroup[bgroup.obs['cell_keep']]
    tempobs.append(bgroup_keep)
    for k in check_clusters:
        bgroupk = bgroup.obs[bgroup.obs['leiden'] == k]
        summary = bgroupk.groupby(['tissue'])['cell_keep'].value_counts(normalize=True).unstack(fill_value=0)
        summary

    print("Cluster 2")
    tempobs[0].obs['cell_keep'].sum()
    print("Cluster 4")
    tempobs[1].obs['cell_keep'].sum()
    print("Cluster 2 and 4")
    tempobs[2].obs['cell_keep'].sum()


# Generate maps of clusters --> manual lineages
manual_mapping = {"0": "Epithelial", "1": "T", "2": "B", "3": "Myeloid", "4": "B", "5": "Mesenchymal", "6": "Epithelial", "7": "Mast", "8": "Platelet+RBC"}

# Add to dataframe and rename the column
adata.obs['manual_lineage'] = adata.obs['leiden'].map(manual_mapping)
adata = adata[adata.obs['manual_lineage'] != "chuck"]
adata.obs = adata.obs.rename(columns={"leiden": f"leiden_{resolution}_round1_QC"}) 
if "cluster" in adata.obs.columns:
    adata.obs.drop(columns="cluster", inplace=True)

# Also remove the previous keep/remove columns
columns_to_remove = [col for col in adata.obs.columns if 'keep' in col]
adata.obs.drop(columns=columns_to_remove, inplace=True)

# Plot the manual lineage
sc.settings.figdir=f"results/{tissue}/figures/UMAP/annotation"
sc.pl.umap(adata, color="manual_lineage", legend_loc="on data", save=f"_{tissue}_manual_lineage_legend_on.png")

# Remove the embeddings, NN, re-count the expression data
adata.obsm.clear()
adata.obsp.clear()
adata.X = adata.layers['counts']
adata.varm.clear()
var_cols_to_clear = ['n_cells', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'mean', 'std']
adata.var.drop(columns=var_cols_to_clear, inplace=True)
adata.uns.clear()

# Re-index
if "cell" in adata.obs.columns.values:
    adata.obs.set_index("cell", inplace=True)

# Print the number of samples and CD samples from each tissue
tissues = np.unique(adata.obs['tissue'])
temp = adata.obs
for t in tissues:
     tempt = temp[temp['tissue'] == t]
     print(f"For {t}, there is {len(np.unique(tempt['sanger_sample_id']))} samples. A total of {tempt.shape[0]} cells")
     cd = tempt[tempt['disease_status'] == "CD"]
     print(f"{len(np.unique(cd['sanger_sample_id']))} are CD")
     print(f"From a total of {len(np.unique(tempt['patient_id']))} individuals")
    

print(f"The final number of individuals is: {len(np.unique(adata.obs['patient_id']))}")


# Save into the results dir
adata.write_h5ad(f"results/{tissue}/objects/adata_manual_lineage_clean.h5ad")

# Dvide by manual lineage and save
lins = np.unique(adata.obs['manual_lineage'])
for l in lins:
    print(l)
    temp = adata[adata.obs['manual_lineage'] == l]
    print(temp.shape)
    temp.write_h5ad(f"input_cluster_within_lineage/adata_manual_lineage_clean_{tissue}_{l}.h5ad")

# Have a look at the variation per lineage at different nMAD thresholds. How does this compare? 
lineages = np.unique(adata.obs['manual_lineage'])
lineages = lineages[~np.isin(lineages,["Mast", "Platelet+RBC"])]
mads_test = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
linobs = []
for l in lineages:
    print(l)
    obs = adata.obs[adata.obs['manual_lineage'] == l]
    linobs.append(obs)
    
    
mads_per_val = []
for mad_val in mads_test:
    print(f"***************** {mad_val} ************")
    mad_sum=[]
    for index, l in enumerate(lineages):
        print(f"********************* {l} *********************")
        obs = linobs[index]
        ladata = ad.AnnData(obs=obs)
        ladata.obs['log_n_genes_by_counts'] = np.log10(ladata.obs['n_genes_by_counts'])
        ladata.obs['log_total_counts'] = np.log10(ladata.obs['total_counts'])
        per_c = []
        for c in cols:
            print(f"~~~~~~ {c} ~~~~~~")
            data = ladata.obs[c].values
            absolute_diff = np.abs(data - np.median(data))
            mad = np.median(absolute_diff)
            cutoff_low = np.median(data) - (mad_val*mad)
            cutoff_high = np.median(data) + (mad_val*mad)
            range = cutoff_high - cutoff_low
            colsum = pd.DataFrame({"lineage": [l], "col": [c], "median": [np.median(data)], "mad": [mad], "cutoff_low":[cutoff_low], "cutoff_high":[cutoff_high], "range":[cutoff_high-cutoff_low], "nMAD": [mad_val]})
            per_c.append(colsum)
        all_c = pd.concat(per_c)
        mad_sum.append(all_c)
    mad_sum_single_mad = pd.concat(mad_sum)
    mads_per_val.append(mad_sum_single_mad)

mad_sum_all = pd.concat(mads_per_val)
# Replace negative values, these are unusable
#mad_sum_all['cutoff_low'] = mad_sum_all['cutoff_low'].where(mad_sum_all['cutoff_low'] >= 0, 0)

# Plot
mad_sum_all.reset_index(drop=True, inplace=True)

plt.rcParams.update({
    'axes.titlesize': 14,    # Title size
    'axes.labelsize': 14,    # Axis label size
    'xtick.labelsize': 12,   # X-tick label size
    'ytick.labelsize': 12,   # Y-tick label size
    'legend.fontsize': 12,   # Legend font size
    'figure.figsize': [8, 16]  # Adjust figure size to be wider and shorter
})


g = sns.FacetGrid(mad_sum_all, row='lineage', col='col', margin_titles=True, despine=False, height=2, aspect=2.5, sharex=False)
def forest_plot(data, **kwargs):
    for _, row in data.iterrows():
        if row['cutoff_low'] < 0:
            lower_err = row['median']
        else:
            lower_err = row['cutoff_high'] - row['median']
        plt.errorbar(
            row['median'], row['nMAD'],
            xerr=[[lower_err], [row['cutoff_high'] - row['median']]],
            fmt='o', color='black'
        )

g.map_dataframe(forest_plot)
g.set_axis_labels("Parameter", "Lineage")
g.set_titles(row_template="{row_name}", col_template="{col_name}")    
for col_val in mad_sum_all['col'].unique():
    col_axes = g.axes[:, mad_sum_all['col'].unique().tolist().index(col_val)]
    col_data = mad_sum_all[mad_sum_all['col'] == col_val]
    x_min = max(min(col_data['cutoff_low']), 0)
    x_max = max(col_data['cutoff_high'])
    
    # Set x-axis limits for all rows in this column
    for ax in col_axes:
        ax.set_xlim(x_min, x_max)


g.tight_layout()
plt.savefig(f"results_round1/{tissue}/figures/UMAP/annotation/mad_range_per_param_per_lineage", bbox_inches='tight')
plt.clf()

# What is most comparable to the mean nMAD spread for non-epi 
imm3 = mad_sum_all[mad_sum_all['nMAD'] == 3]
imm3 = imm3[imm3['lineage'].isin(["B", "T", "Myeloid"]) ]
# Find mean range of each val
mean_cols = {}
for c in cols:
    mean_cols[c] = imm3.loc[imm3['col'] == c, 'range'].values.sum()/len(np.unique(imm3['lineage']))

# For each range, find the difference for each mad of the epi nMADs
epi = mad_sum_all[mad_sum_all['lineage'] == "Epithelial"]
dist_imm3 = []
for c in cols:
    temp = epi[epi['col'] == c]
    temp['dist_to_imm3'] = temp['range'] - mean_cols[c]
    dist_imm3.append(temp)

# Put all together and save
dist_imm3_all = pd.concat(dist_imm3)
dist_imm3_all.to_csv(f"results_round1/{tissue}/figures/UMAP/annotation/epi_range_distance_to_imm_range.csv")