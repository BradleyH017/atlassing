#!/usr/bin/env python
####################################################################################################################################
####################################################### Bradley April 2024 #########################################################
##################################### Summary script from the within lineage clustering ############################################
###### To be ran after the second round of the analysis (Snakefile_cluster_within_lineage.smk, config_cluster_within_lineage) ######
####################################################################################################################################
####################################################################################################################################

# Import libraries
import scanpy as sc
import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad

# Define options
tissue="rectum"
lineages = ["epithelial", "immune", "mesenchymal"]

# Propose cut offs
max_proportion = 0.25
min_ncells = 40
min_MCC=0.75

# For each lineage and each resolution, load in the anndata, append MCC, and do cluster QC
max_resolutions = []
combined_adatas = []
prop_sums = []
for l in lineages:
    print(f"~~~~~~~~~~~~ Lineage: {l}~~~~~~~~~~~~~")
    adata = sc.read_h5ad(f"results/{tissue}_{l}/objects/adata_PCAd_batched_umap.h5ad")
    adata.obs.reset_index(inplace=True)
    file_paths = glob.glob(f"results/{tissue}_{l}/tables/clustering_array/leiden_*")
    mcc_all = []
    prop_sum_filtered_all = []
    for f in file_paths:
        cl = pd.read_csv(f"{f}/clusters.csv")
        res = cl.columns[1]
        print(res)
        adata.obs = adata.obs.merge(cl, on="cell", how="left")
        mcc = pd.read_csv(f"{f}/base-model_report.tsv.gz", compression="gzip", sep = "\t")
        mcc = mcc[mcc['is_cluster'] == True]
        mcc_all.append(mcc)
        mcc[res] = mcc['cell_label']
        to_add = mcc[['MCC', "n_cells_full_dataset", res]]
        #to_add = to_add.rename(columns={"MCC": f"MCC_{res}", "n_cells_full_dataset": f"n_cells_full_dataset_{res}"})
        adata.obs[res] = adata.obs[res].astype(str)
        #adata.obs = adata.obs.merge(to_add, on=res, how="left")
        #
        # Summarise the maximum proportion of each cluster contributed to by each sample
        proportions = adata.obs.groupby(res)['samp_tissue'].value_counts(normalize=True).reset_index(name="samp_tissue_proportion")
        max_index = proportions.groupby(res)["samp_tissue_proportion"].idxmax()
        prop_sum = proportions.loc[max_index]
        prop_sum = prop_sum.merge(to_add, on=res, how="left")
        #
        # Plot these, with proposed cut offs on
        # Max single sample
        plt.figure(figsize=(8, 6))
        # Plot points that pass
        plt.scatter(prop_sum.loc[(prop_sum['MCC'].astype(float) > min_MCC) & 
                                (prop_sum['samp_tissue_proportion'] < max_proportion),
                                'samp_tissue_proportion'],
                    prop_sum.loc[(prop_sum['MCC'].astype(float) > min_MCC) & 
                                (prop_sum['samp_tissue_proportion'] < max_proportion),
                                'MCC'].astype(float),
                    s=100, color='orange', label='Pass cluster QC')
        #
        # Scatter plot for other points (points will be blue)
        plt.scatter(prop_sum.loc[(prop_sum['MCC'].astype(float) <= min_MCC) | 
                                (prop_sum['samp_tissue_proportion'] >= max_proportion),
                                'samp_tissue_proportion'],
                    prop_sum.loc[(prop_sum['MCC'].astype(float) <= min_MCC) | 
                                (prop_sum['samp_tissue_proportion'] >= max_proportion),
                                'MCC'].astype(float),
                    s=100, color='lightblue', label='Fail cluster QC')
        #
        for i, txt in enumerate(prop_sum[res]):
            plt.text(prop_sum['samp_tissue_proportion'][i], prop_sum['MCC'][i], str(txt), ha='center', va='center')
        #
        plt.title(f"{tissue} - {res} cluster QC - single_samp")
        plt.axvline(x = max_proportion, color = 'black', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_MCC, color = 'black', linestyle = '--', alpha = 0.5)
        plt.legend()
        plt.xlabel('Max proportion from a single sample')
        plt.ylabel('MCC')
        plt.savefig(f"{f}/MCC_vs_max_single_sample_{res}.png", bbox_inches='tight')
        plt.clf()
        #
        # Min cells
        plt.figure(figsize=(8, 6))
        # Plot points that pass
        plt.scatter(prop_sum.loc[(prop_sum['MCC'].astype(float) > min_MCC) & 
                                (prop_sum['n_cells_full_dataset'] > min_ncells),
                                'n_cells_full_dataset'],
                    prop_sum.loc[(prop_sum['MCC'].astype(float) > min_MCC) & 
                                (prop_sum['n_cells_full_dataset'] > min_ncells),
                                'MCC'].astype(float),
                    s=100, color='orange', label='Pass cluster QC')
        #
        # Scatter plot for other points (points will be blue)
        plt.scatter(prop_sum.loc[(prop_sum['MCC'].astype(float) <= min_MCC) | 
                                (prop_sum['n_cells_full_dataset'] <= min_ncells),
                                'n_cells_full_dataset'],
                    prop_sum.loc[(prop_sum['MCC'].astype(float) <= min_MCC) | 
                                (prop_sum['n_cells_full_dataset'] <= min_ncells),
                                'MCC'].astype(float),
                    s=100, color='lightblue', label='Fail cluster QC')
        for i, txt in enumerate(prop_sum[res]):
            plt.text(prop_sum['n_cells_full_dataset'][i], prop_sum['MCC'][i], str(txt), ha='center', va='bottom')
        #
        plt.title(f"{tissue} - {res} cluster QC - ncells")
        plt.axvline(x = min_ncells, color = 'black', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_MCC, color = 'black', linestyle = '--', alpha = 0.5)
        plt.xlabel('Number of cells in full dataset')
        plt.ylabel('MCC')
        plt.savefig(f"{f}/MCC_vs_ncells_{res}.png", bbox_inches='tight')
        plt.clf()
        #
        # If we were to apply these filters, which clusters would we keep?
        prop_sum_filtered = prop_sum[(prop_sum['n_cells_full_dataset'] > min_ncells) & (prop_sum['samp_tissue_proportion'] < max_proportion)]
        prop_sum_filtered['all_MCC_pass'] = all(prop_sum_filtered['MCC'] > min_MCC)
        prop_sum_filtered['res'] = float(res.strip("leiden_"))
        prop_sum_filtered = prop_sum_filtered.rename(columns={res: "cell_label"})
        prop_sum_filtered_all.append(prop_sum_filtered)
        #
    prop_sum_filtered_all = pd.concat(prop_sum_filtered_all)
    prop_sums.append(prop_sum_filtered_all)
    # Find the max res
    use_res = max(prop_sum_filtered_all.loc[prop_sum_filtered_all['all_MCC_pass'] == True,'res'])
    print(f"~~~~~~~~~~~~~~~~~~~~~~~~ For the {l} lineage, the maximum res after cluster QC is {use_res} ~~~~~~~~~~~~~~~~~~~~~~~~")
    prop_sum_filtered_all = prop_sum_filtered_all[prop_sum_filtered_all['res'] == use_res]
    print(prop_sum_filtered_all)
    max_resolutions.append(use_res)
    # Subset the adata to include these reproducible cells passing these thresholds and put together
    adata.obs['final_clusters'] = adata.obs[f'leiden_{str(use_res)}']
    adata = adata[adata.obs['final_clusters'].isin(prop_sum_filtered_all['cell_label'])]
    cols_to_keep = [col for col in adata.obs.columns if not col.startswith('leiden_')]
    # Create a new DataFrame with only the columns that do not start with "leiden_"
    adata.obs = adata.obs[cols_to_keep]
    adata.obs['final_clusters'] = adata.obs['final_clusters'].apply(lambda x: f'{l}_{x}')
    adata.obs = adata.obs.rename(columns={"final_clusters": "leiden"})
    combined_adatas.append(adata)
    del(adata)


adata = ad.concat(combined_adatas)


# Combine these clusters with that of the round 1 embedding, then perform celltypist on this and compare
add_clusters = adata.obs[['cell', 'leiden']]
del adata
round1_adata = sc.read_h5ad(f"results_round1/{tissue}/objects/adata_PCAd_batched_umap.h5ad")
orig = round1_adata.shape[0]
new = add_clusters.shape[0]
print(f"After cluster filtration, lose {str(orig - new)} cells")
round1_adata = round1_adata[round1_adata.obs.index.isin(add_clusters['cell'])]
round1_adata.obs.reset_index(inplace=True)
round1_adata.obs = round1_adata.obs.merge(add_clusters, on="cell", how="left")
round1_adata.obs.set_index("cell", inplace=True)

print(f"round1_adata final shape: {round1_adata.shape}")
round1_adata.write_h5ad("results_round1/rectum/objects/adata_cross_lineage_refined_clusters.h5ad")

    