########### Bradley June 2024 #########
import scanpy as sc
import pandas as pd
import numpy as np
import os
from anndata.experimental import read_elem
from h5py import File
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
os.chdir("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined")

# Define lineage and resolution
lineage = "all_immune"
res = 1.0
max_prop = 0.2
MCC_thresh = 0.75

# Read in adata obs
f = f'results/{lineage}/tables/clustering_array/leiden_{str(res)}/adata_PCAd_batched_umap_{str(res)}.h5ad'
f2 = File(f, 'r')
# Read only cell-metadata
obs = read_elem(f2['obs'])

# find exceeding sample
proportions = obs.groupby('leiden')['samp_tissue'].value_counts(normalize=True).reset_index(name="samp_tissue_proportion")
exceeding_samples = proportions[proportions['samp_tissue_proportion'] > max_prop]
exceeding_samples = np.unique(exceeding_samples['samp_tissue'])

# Have a look at this sample on the initial sample depth_count statistics
depth_count = pd.read_csv(f"results/{lineage}/tables/depth_count_pre_cell_filtration.csv", index_col=0)

# Plot, highlighting this sample
min_median_nCount_per_samp_blood = 0
min_median_nGene_per_samp_blood = 1000
min_median_nCount_per_samp_gut = 0
min_median_nGene_per_samp_gut = 1000
pathout = f"results/{lineage}/tables/clustering_array/leiden_{str(res)}"
depth_count['exceeding_samples'] = depth_count.index.map(lambda x: 'red' if x in exceeding_samples else 'navy')

td = np.unique(depth_count['disease_tissue'])
for t in td:
    samps = obs[obs['disease_tissue'] == t]['samp_tissue']
    plt.figure(figsize=(8, 6))
    plt.scatter(depth_count[depth_count.index.isin(samps)]["Median_nCounts"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["exceeding_samples"], alpha=0.7)
    plt.xlabel('Median counts / cell')
    plt.ylabel('Median genes detected / cell')
    if t == "blood":
        plt.axvline(x = min_median_nCount_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_median_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
        plt.title(f"{t} - min_median_nCount: {min_median_nCount_per_samp_blood}, min_median_nGene: {min_median_nGene_per_samp_blood}")
    else:
        plt.axvline(x = min_median_nCount_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
        plt.title(f"{t} - min_median_nCount: {min_median_nCount_per_samp_gut}, min_median_nGene: {min_median_nGene_per_samp_gut}")
    
    plt.savefig(f"{pathout}/sample_median_counts_ngenes_{t}.png", bbox_inches='tight')
    plt.clf()
    
# Do same for MT percentage
for t in td:
    samps = obs[obs['disease_tissue'] == t]['samp_tissue']
    plt.figure(figsize=(8, 6))
    plt.scatter(depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"], depth_count[depth_count.index.isin(samps)]["Median_MT"],  c=depth_count[depth_count.index.isin(samps)]["exceeding_samples"], alpha=0.7)
    plt.xlabel('Median genes detected / sample')
    plt.ylabel('Median MT% / sample')
    plt.title(f"{t}")
    plt.savefig(f"{pathout}/sample_median_ngene_mt_{t}.png", bbox_inches='tight')
    plt.clf()

# Colour the PCA by tissue x disease
pca = pd.read_csv(f"{pathout}/pca_cell_contributions_full_df.csv")
depth_count.reset_index(inplace=True)
pca = pca.merge(depth_count[['samp_tissue', "disease_tissue", "exceeding_samples"]], on = "samp_tissue", how="left")
palette = sns.color_palette("muted", len(pca['disease_tissue'].unique()))
color_dict = {category: palette[i] for i, category in enumerate(pca['disease_tissue'].unique())}
plt.figure(figsize=(10, 8))
for category, color in color_dict.items():
    subset = pca[pca['disease_tissue'] == category]
    plt.scatter(subset['PC1'], subset['PC2'], label=category, alpha=0.6, c=[color])


plt.xlabel('Cell contribution PC 1')
plt.ylabel('Cell contribution PC 1')
plt.title(f'PCA of cluster proportions')
plt.legend(title='Tissue Disease')
plt.savefig(f"{pathout}/pca_cell_contributions_disease_tissue.png", bbox_inches='tight')

# Have a look at the cell contributionss of CD blood only, and where this sample lies? 
contr_cl = obs.groupby('samp_tissue')['leiden'].value_counts(normalize=True).reset_index(name="samp_tissue_contribution")
test_td = depth_count[depth_count['samp_tissue'].isin(exceeding_samples)]
test_td = np.unique(test_td['disease_tissue'])
contr_cl = obs.groupby('samp_tissue')['leiden'].value_counts(normalize=True).reset_index(name="samp_tissue_contribution")
# Make this square
pivoted_df = contr_cl.pivot(index='samp_tissue', columns='leiden', values='samp_tissue_contribution').fillna(0)
# Plot PCA for just these samples
for t in test_td:
    samps = depth_count[depth_count['disease_tissue'] == t]
    samps = samps['samp_tissue']
    subset = pivoted_df[pivoted_df.index.isin(samps)]
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(subset)
    pca_df = pd.DataFrame(data=pca_result[:,:2], columns=['PC1', 'PC2'], index=subset.index)
    pca_df['exceeding_sample'] = pca_df.index.map(lambda x: 'exceeding_sample' if x in exceeding_samples else 'okay')
    palette = sns.color_palette("muted", len(pca_df['exceeding_sample'].unique()))
    color_dict = {category: palette[i] for i, category in enumerate(pca_df['exceeding_sample'].unique())}
    plt.figure(figsize=(10, 8))
    for category, color in color_dict.items():
        subset = pca_df[pca_df['exceeding_sample'] == category]
        plt.scatter(subset['PC1'], subset['PC2'], label=category, alpha=0.6, c=[color])
    #
    plt.xlabel('Cell contribution PC 1')
    plt.ylabel('Cell contribution PC 1')
    plt.title(f'PCA of cluster proportions')
    plt.legend(title=t)
    plt.savefig(f"{pathout}/pca_cell_contributions_{t}_exceeding_sample.png", bbox_inches='tight')
    
# Highlight cluster QC on the basis of this cluster:
annots = pd.unique(obs['leiden'])
annots = np.sort(annots.astype(int)).astype(str)
failing = np.unique(proportions[proportions['samp_tissue_proportion'] > max_prop].leiden)
passing = np.setdiff1d(annots, failing)
qc_cols = ['log10_total_counts', 'total_counts_nMAD', 'log10_n_genes_by_counts', 'n_genes_by_counts_nMAD', 'MT_perc_nMads', 'pct_counts_gene_group__mito_transcript']
for qc in qc_cols:
    print(f"Plotting {qc} per cluster - exceeding aware")
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    for a in annots:
        data = obs[obs['leiden'] == a]
        data = data[qc]
        if a in passing:
            sns.distplot(data, hist=False, rug=True, label=f"{a} (samp_prop < {max_prop})", kde_kws={'color': 'orange'})
        else:
            sns.distplot(data, hist=False, rug=True, label=a)
    
    plt.legend(loc='upper center', bbox_to_anchor=(1, 1), ncol=1)
    plt.xlabel(qc)
    plt.title(f"Distribution of {qc} - {res}: samp_prop")
    plt.savefig(f"{pathout}/{qc}_per_cluster_samp_prop.png", bbox_inches='tight')
    plt.clf()
    
# What does megagut call this cluster? 
test_fail = obs[obs['leiden'].isin(failing)]
ct_pred = test_fail['Celltypist:megagut_celltypist_lowerGI+lym_adult_mar24:predicted_labels'].value_counts(normalize=True)
ct_pred.to_csv(f"{pathout}/celltypist_proportions_exceeding_sample_cluster.csv")

# How exclusive is this annotation to this cluster? 
top = ct_pred.index[0]
ct_prop_across_all = obs.groupby('Celltypist:megagut_celltypist_lowerGI+lym_adult_mar24:predicted_labels')['leiden'].value_counts(normalize=True).reset_index(name="proportion")
ct_prop_test = ct_prop_across_all[ct_prop_across_all['Celltypist:megagut_celltypist_lowerGI+lym_adult_mar24:predicted_labels'] == top]
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(ct_prop_test['leiden'], ct_prop_test['proportion'], color='skyblue')
ax.set_xlabel('leiden')
ax.set_ylabel('Proportion')
ax.set_title(f"Share of CellTypist {top} annotations across clusters")
ax.set_xticks(ct_prop_test['leiden'])
plt.savefig(f"{pathout}/share_of_exceeding_sample_prop_cluster_across_all.png", bbox_inches='tight')

# have a look at metadata that contributes to the really failing cluster
mcc = pd.read_csv(f"{pathout}/base-model_report.tsv.gz", compression="gzip", sep = "\t")
mcc = mcc[mcc['is_cluster'] == True]
mcc['resolution'] = res 
failing_mcc = mcc[mcc['MCC'] < MCC_thresh]
metadata_to_check = ['sex', 'age', 'disease_status', 'inflammation_status']
fullmeta=pd.read_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/andersonlab-select_samples_atlas/metadata_combined/metadata_combined.tsv", sep = "\t")
fullmeta['samp_tissue'] = fullmeta["sanger_sample_id"] + "_" + fullmeta['biopsy_type']
to_add = fullmeta[["ti_neoti", "f2_f1_ratio", "smoking_status", "medication", "samp_tissue"]]
obs_samp = obs[metadata_to_check + ['leiden', 'samp_tissue']].reset_index()
obs_samp = obs_samp[metadata_to_check + ['leiden', 'samp_tissue']].drop_duplicates()
obs_samp = obs_samp.merge(to_add, on="samp_tissue", how="left")
for i, column in enumerate(metadata_to_check + ["ti_neoti", "f2_f1_ratio", "smoking_status", "medication"]):
    grouped = obs_samp.groupby('leiden')[column].value_counts().unstack().fillna(0)
    proportions = grouped.div(grouped.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    proportions.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_ylabel('Proportion')
    ax.set_title(f'Proportion of {column} across leiden')
    ax.legend(title=column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"{pathout}/distribution_of_{column}_across_clusters.png", bbox_inches='tight')
