#!/usr/bin/env python
####################################################################################################################################
####################################################### Bradley April 2024 #########################################################
##################################### Summary script from the within lineage clustering ############################################
###### To be ran after the second round of the analysis (Snakefile_cluster_within_lineage.smk, config_cluster_within_lineage) ######
####################################################################################################################################
####################################################################################################################################
# This script filters clusters and badly integrating samples on the basis of the proportion contributed by a single sample. 
# This also has an option for removing failing clusters (after this removal) on the basis of whether they contribute heaavily to clusters failing at lower resolutions


# Import libraries
import scanpy as sc
import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
from sklearn.decomposition import PCA
import plotnine as plt9
print("Loaded packages")

# Define plot function
def _make_plots(
    df_plt,
    out_file_base,
    y='AUC',
    facet_grid='',
    h_line=''
):
    len_x = len(np.unique(df_plt['resolution']))
    if 'sparsity_l1' in df_plt.columns:
        df_plt['Sparsity'] = df_plt['sparsity_l1']
        len_x2 = len(np.unique(df_plt['Sparsity']))
    else:
        len_x2 = 0
    if len_x2 > 1:
        gplt = plt9.ggplot(df_plt, plt9.aes(
            fill='Sparsity',
            x='resolution',
            y=y,
        ))
        gplt = gplt + plt9.geom_boxplot(
            alpha=0.8,
            outlier_alpha=0
        )
        gplt = gplt + plt9.geom_jitter(
            plt9.aes(color='Sparsity'),
            alpha=0.25,
            width=0.2
        )
    else:
        gplt = plt9.ggplot(df_plt, plt9.aes(
            x='resolution',
            y=y
        ))
        gplt = gplt + plt9.geom_boxplot(
            alpha=0.8,
            outlier_alpha=0
        )
        gplt = gplt + plt9.geom_jitter(
            alpha=0.25,
            width=0.2
        )
    gplt = gplt + plt9.theme_bw(base_size=12)
    if facet_grid != '':
        gplt = gplt + plt9.facet_grid('{} ~ .'.format(facet_grid))
    if y == 'f1-score':
        gplt = gplt + plt9.labs(
            x='Resolution',
            y='F1 score',
            title=''
        )
    elif y in ['AUC', 'MCC']:
        gplt = gplt + plt9.labs(
            x='Resolution',
            y=y,
            title=''
        )
    else:
        gplt = gplt + plt9.labs(
            x='Resolution',
            y=y.capitalize().replace('_', ' '),
            title=''
        )
    gplt = gplt + plt9.theme(
        # legend_position='none',
        axis_text_x=plt9.element_text(angle=-45, hjust=0)
    )
    if len_x2 != 0 and len_x2 < 9:
        gplt = gplt + plt9.scale_fill_brewer(
            palette='Dark2',
            type='qual'
        )
    if h_line != '':
        gplt = gplt + plt9.geom_hline(
            plt9.aes(yintercept=h_line),
            linetype='dashdot'
        )
    gplt.save(
        '{}-resolution__{}.png'.format(out_file_base, y.replace('-', '_')),
        #dpi=300,
        width=4*((len_x+len_x2)/4),
        height=5,
        limitsize=False
    )

def main():
    # if testing
    #tissue="all"
    #within_lineages = True
    #max_proportion = 0.2
    #min_ncells = 0
    #min_MCC=0.75
    #ilter_based_on_prev_fails = True
    #consistent_failing_cluster_proportion=0.8
    #num_consistent_failing_clusters=1
    
    inherited_options = parse_options()
    # REST
    
    
    # Define lineages
    if within_lineages:
        lineages = ["epithelial", "immune", "mesenchymal"]
    else:
        lineages = [""]

    # For each lineage and each resolution, load in the anndata, append MCC, and do cluster QC
    max_resolutions = []
    combined_adatas = []
    prop_sums = []
    blacklist_samples_all = []

    # Define function to reorder
    def get_numeric_suffix(column_name):
        return float(column_name.split("_")[-1])

    for l in lineages:    
        print(f"~~~~~~~~~~~~ Lineage: {l}~~~~~~~~~~~~~")
        if within_lineages:
            adata = sc.read_h5ad(f"results/{tissue}_{l}/objects/adata_PCAd_batched_umap.h5ad")
        else:
            adata = sc.read_h5ad(f"results/{tissue}/objects/adata_PCAd_batched_umap.h5ad")
        #
        adata.obs.reset_index(inplace=True)
        if within_lineages:
            file_paths = glob.glob(f"results/{tissue}_{l}/tables/clustering_array/leiden_*")
        else:
            file_paths = glob.glob(f"results/{tissue}/tables/clustering_array/leiden_*")
        #
        # Order file_paths:
        file_paths = sorted(file_paths, key=lambda x: float(x.split("leiden_")[1]))
        mcc_all = []
        prop_sum_filtered_all = []
        black_list_samples = []
        all_clusters = []
        for f in file_paths:
            cl = pd.read_csv(f"{f}/clusters.csv")
            if len(all_clusters) == 0:
                all_clusters = cl
            else:
                all_clusters = all_clusters.merge(cl, on="cell", how="left")
                sorted_columns = ["cell"] + sorted(all_clusters.columns[1:], key=get_numeric_suffix)
                all_clusters = all_clusters.reindex(columns=sorted_columns)
            #
            res = cl.columns[1]
            print(res)
            adata.obs = adata.obs.merge(cl, on="cell", how="left")
            mcc = pd.read_csv(f"{f}/base-model_report.tsv.gz", compression="gzip", sep = "\t")
            mcc = mcc[mcc['is_cluster'] == True]
            if len(mcc_all) == 0:
                mcc_all = mcc
            else:
                mcc_all = pd.concat([mcc_all,mcc], ignore_index=True)
            #
            #mcc_all.append(mcc)
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
            # Also plot a stacked bar for the number of cells in each cluster for all sample
            cl_counts= adata.obs.groupby(res)['samp_tissue'].value_counts().reset_index()
            total_counts = cl_counts.groupby('samp_tissue')['count'].transform('sum')
            cl_counts['total_count'] = total_counts
            fig, ax = plt.subplots(figsize=(12, 8))
            for key, grp in cl_counts.groupby(res):
                ax.bar(grp['samp_tissue'], grp['count'], label=key, alpha=0.7)
            #
            ax.set_ylabel('Count')
            ax.set_xlabel('Sample Tissue')
            ax.set_title(f'Stacked Bar Chart of Count by {res}')
            ax.legend(title=res, bbox_to_anchor=(1, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.savefig(f"{f}/counts_per_sample_{res}.png", bbox_inches='tight')
            plt.clf()
            #
            # if a sample contributes really heavily to a single cluster (which is removed by out cut off), does that sample contribute considerable to other clusters)
            bad_samples = prop_sum.loc[prop_sum['samp_tissue_proportion'] > max_proportion, "samp_tissue"].values
            black_list_df = pd.DataFrame({"res": res, "samp_tissue": bad_samples})
            bl_samp_counts = adata.obs.loc[adata.obs['samp_tissue'].isin(bad_samples), 'samp_tissue'].value_counts().reset_index()
            black_list_df = black_list_df.merge(bl_samp_counts, on="samp_tissue", how="left")
            # Add this to the blacklist of samples
            black_list_samples.append(black_list_df)
            if len(bad_samples) > 0:
                # Make a plot showing proportion contributed to each cluster by that sample
                for b in bad_samples:
                    bad_props = proportions[proportions['samp_tissue'] == b]
                    bad_samp_cluster = prop_sum.loc[(prop_sum['samp_tissue'] == b) & (prop_sum['samp_tissue_proportion'] > max_proportion), res].values
                    bad_props[res] = pd.to_numeric(bad_props[res])
                    bad_props = bad_props.sort_values(by=res, ascending=True)
                    ncells = black_list_df.loc[black_list_df['samp_tissue'] == b, "count"].values[0]
                    print(f"Plotting bad sample {b} / cluster {bad_samp_cluster[0]}")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(bad_props[res], bad_props['samp_tissue_proportion'], color='skyblue')
                    ax.set_xlabel(res)
                    ax.set_ylabel('Sample Tissue Proportion')
                    ax.set_title(f"Contribution of {b} (max prop cluster {bad_samp_cluster[0]}) to others (total n={ncells})")
                    ax.set_xticks(bad_props[res])
                    plt.savefig(f"{f}/exceeding_sample_prop_cluster_{bad_samp_cluster[0]}_{res}.png", bbox_inches='tight')
                    plt.clf()
                    # Also plot the contribution of all samples to this cluster
                    prop_each_sample_bad_cluster = proportions[proportions[res] == bad_samp_cluster[0]]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(prop_each_sample_bad_cluster['samp_tissue'], prop_each_sample_bad_cluster['samp_tissue_proportion'], color='skyblue')
                    ax.set_xlabel('sample')
                    ax.set_ylabel('Relative contribution of each sample')
                    ax.set_title(f"Contribution of each sample to cluster {bad_samp_cluster[0]}")
                    plt.xticks(rotation=45, ha='right')
                    plt.savefig(f"{f}/potential_bad_cluster_{bad_samp_cluster[0]}_{res}.png", bbox_inches='tight')
                    plt.clf()
                    # Print the samples with contributions to this cluster over the cut off
                    offending_samps = ', '.join(list(prop_each_sample_bad_cluster.loc[prop_each_sample_bad_cluster["samp_tissue_proportion"] > max_proportion, "samp_tissue"].astype(str)))
                    print(f"Samples with contribution > {max_proportion} to cluster {bad_samp_cluster[0]} are {offending_samps}")
            #
            #
            # For the bad samples, calculate the proportion of their cells to all clusters
            contr_cl = adata.obs.groupby('samp_tissue')[res].value_counts(normalize=True).reset_index(name="samp_tissue_contribution")
            # Make this square
            pivoted_df = contr_cl.pivot(index='samp_tissue', columns=res, values='samp_tissue_contribution').fillna(0)
            if len(np.unique(contr_cl[res])) > 10:
                n_components = 10
            else:
                n_components=2
            #
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(pivoted_df)
            pca_df = pd.DataFrame(data=pca_result[:,:2], columns=['PC1', 'PC2'], index=pivoted_df.index)
            plt.figure(figsize=(10, 8))
            plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
            #
            # Highlight bad samples
            for sample in bad_samples:
                if sample in pca_df.index:
                    plt.scatter(pca_df.loc[sample, 'PC1'], pca_df.loc[sample, 'PC2'], color='red', label='Bad Sample')
                    plt.annotate(sample, (pca_df.loc[sample, 'PC1'], pca_df.loc[sample, 'PC2']))
            #
            plt.xlabel('Cell contribution PC 1')
            plt.ylabel('Cell contribution PC 1')
            plt.title('PCA of the relative contribution of each sample to each cluster')
            plt.savefig(f"{f}/pca_cell_contributions.png", bbox_inches='tight')
            plt.clf()        
            # If we were to apply these filters, which clusters would we keep?
            prop_sum_filtered = prop_sum[(prop_sum['n_cells_full_dataset'] > min_ncells) & (prop_sum['samp_tissue_proportion'] < max_proportion)] # min ncells and max samp proportion
            if filter_based_on_prev_fails:
                # Have a look at whether there is one or two failing clusters, and if those cells are frequently failing previous resolutions. 
                # If this is the case, can make an argument for renmoving this cluster only and keeping the rest
                still_fails = (prop_sum_filtered['MCC'] < min_MCC).sum()
                if (still_fails <= num_consistent_failing_clusters) & (still_fails > 0):
                    still_fails_in_other_clusters = []
                    final_remove_clusters = []
                    print("There is still failing clusters, having a look at these in other clusters")
                    still_fails_clusters = prop_sum_filtered.loc[prop_sum_filtered['MCC'] < min_MCC, res].values.astype(int)
                    for c in still_fails_clusters:
                        # Find where these cells are in previous resolutions, are they still failing? 
                        still_fails_cluster = int(prop_sum_filtered.loc[prop_sum_filtered['MCC'] < min_MCC, res].values[0])
                        still_fails_cluster_cells = all_clusters.loc[all_clusters[res] == still_fails_cluster, "cell"].values
                        # What is the contribution of these cells to other clusters in the lower 2 resolutions
                        prev_2_res = all_clusters.columns[[-3,-2]].values
                        prev_2_res = [column for column in prev_2_res if column != 'cell']
                        still_fails_in_other_clusters = []
                        for r in prev_2_res:
                            print(f"Looking at {r}")
                            temp_r = all_clusters[all_clusters['cell'].isin(still_fails_cluster_cells)]
                            cell_counts = temp_r[r].value_counts().reset_index()
                            cell_counts['prop'] = cell_counts['count']/len(still_fails_cluster_cells)
                            cell_counts['prev_res'] = cell_counts[r].map(lambda x: f"{r}-{x}")
                            cell_counts.rename(columns={r: "cell_label"}, inplace=True)
                            # Merge with corresponding MCC
                            mcc_to_add = mcc_all[mcc_all['resolution'] == float(r.strip("leiden_"))]
                            mcc_to_add = mcc_to_add[["cell_label", "MCC"]]
                            mcc_to_add['cell_label'] = mcc_to_add['cell_label'].astype(int)
                            mcc_to_add = mcc_to_add[mcc_to_add['cell_label'].isin(cell_counts['cell_label'])]
                            cell_counts = cell_counts.merge(mcc_to_add, on="cell_label", how="left")
                            cell_counts['resolution'] = r
                            if len(still_fails_in_other_clusters) == 0:
                                still_fails_in_other_clusters = cell_counts
                            else:
                                still_fails_in_other_clusters = pd.concat([still_fails_in_other_clusters, cell_counts])
                        #
                        # Save this to file
                        print(still_fails_in_other_clusters)
                        still_fails_in_other_clusters['failing_cluster'] = f"{res}-{still_fails_cluster}"
                        still_fails_in_other_clusters.to_csv(f"{f}/still_failing_cluster{c}_in_prev_resolutions.csv", index=False)
                        # If there are clusters consistently failing that are being contributed heavily by these cells, remove these cells
                        heavily_contributed = still_fails_in_other_clusters[still_fails_in_other_clusters['prop'] > consistent_failing_cluster_proportion]
                        heavily_contributed_failing = heavily_contributed[heavily_contributed['MCC'] < min_MCC]
                        if heavily_contributed_failing.shape[0] > 0:
                            heavily_contributed_failing.to_csv(f"{f}/still_failing_cluster{c}_in_prev_resolutions_heavily_contributed.csv", index=False)
                            if heavily_contributed_failing.shape[0] == 2:
                                print(f"For {res}: Failing cluster{c} contributes at least {consistent_failing_cluster_proportion*100}% to failing clusters in the previous two resolutions, removing this cluster")
                                final_remove_clusters.append(c)
                    
                    # remove the still failing clusters from consideration for the all clusters pass
                    neg_mask = prop_sum_filtered[res].astype(int).isin(final_remove_clusters)
                    mask = ~neg_mask
                    prop_sum_filtered = prop_sum_filtered[mask]
            #
            # Now add whether all remaining clusters pass or not
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
        # Finally, remove cells from the black list samples (those that had exceeding contribution to a given cluster)
        bad_samples_all = pd.concat(black_list_samples)
        bad_samples_use_res = bad_samples_all.loc[bad_samples_all['res'] == f"leiden_{use_res}", "samp_tissue"].values
        blacklist_samples_all.append(bad_samples_use_res)
        combined_adatas.append(adata)
        del(adata)

    # Combine together
    adata = ad.concat(combined_adatas)

    # Remove cells from samples found to be offending QC in any lineage
    blacklist_samples_combined = []
    for lst in blacklist_samples_all:
        blacklist_samples_combined.extend(lst.astype(str))

    print("Blacklist samples per lineage")
    print(blacklist_samples_combined)
    blacklist_samples_combined = np.unique(blacklist_samples_combined)
    print("Blacklist samples across all")
    print(blacklist_samples_combined)

    adata.obs.set_index("cell", inplace=True)
    adata = adata[~adata.obs['samp_tissue'].isin(blacklist_samples_combined.astype(str))]
    adata.obs.reset_index(inplace=True)

    ######## May wish to manually select resolutions (and exclude samples), overiding the above #########
    overide = False
    if overide:
        #del adata
        manual_adatas = []
        # Create dictionary for resolution to use
        import json
        use_resolutions = {"epithelial": "1.0", "immune": "1.5", "mesenchymal": "1.0"}
        clusters_remove = {"epithelial": ["24", "28", "29"], "immune": ["27", "30", "32"], "mesenchymal": ""}
        blacklist_samples = ["scrnacdb13098576__donor_blood", "5892STDY13265991__donor_ti", "OTARscRNA13669872__donor_ti", "5892STDY11060801__donor_ti"]
        # For each lineage, load in the adata, merge with desired clusters and subset
        for l in lineages:
            print(l)
            adata = sc.read_h5ad(f"results/{tissue}_{l}/objects/adata_PCAd_batched_umap.h5ad")
            clusters = pd.read_csv(f"results/{tissue}_{l}/tables/clustering_array/leiden_{use_resolutions[l]}/clusters.csv")
            adata.obs.reset_index(inplace=True)
            clusters = clusters.rename(columns={f"leiden_{use_resolutions[l]}": "leiden"})
            adata.obs = adata.obs.merge(clusters, on="cell", how="left")
            adata.obs.set_index("cell", inplace=True)
            if len(clusters_remove[l]) > 0:
                adata = adata[~adata.obs['leiden'].astype(str).isin(clusters_remove[l])]
            #
            adata.obs['leiden'] = l + "_" + adata.obs['leiden'].astype(str)
            manual_adatas.append(adata)
            del adata
        #
        # Combine 
        adata = ad.concat(manual_adatas)  
        # Remove blacklist samples manually  
        adata = adata[~adata.obs['samp_tissue'].isin(blacklist_samples)]
        del manual_adatas
        adata.obs.reset_index(inplace=True)

    # Extract cells and annotation
    add_clusters = adata.obs[['cell', 'leiden']]

    # Combine these clusters with that of the round 1 embedding, then perform celltypist on this and compare
    del adata
    del combined_adatas
    round1_adata = sc.read_h5ad(f"results_round1/{tissue}/objects/adata_PCAd_batched_umap.h5ad")
    orig = round1_adata.shape[0]
    new = add_clusters.shape[0]
    if "cell" in round1_adata.obs.columns.values:
        round1_adata.obs.set_index("cell", inplace=True)

    print(f"After cluster filtration, lose {str(orig - new)} cells")
    round1_adata = round1_adata[round1_adata.obs.index.isin(add_clusters['cell'])]
    round1_adata.obs.reset_index(inplace=True)
    round1_adata.obs = round1_adata.obs.merge(add_clusters, on="cell", how="left")
    round1_adata.obs.set_index("cell", inplace=True)

    print(f"round1_adata final shape: {round1_adata.shape}")
    round1_adata.obs['leiden'] = round1_adata.obs['leiden'].astype(str)

    # Make new outdirs to save within THIS results directory
    os.makedirs(f"results/{tissue}/figures/UMAP/annotation")
    os.makedirs(f"results/{tissue}/objects/")
    round1_adata.write_h5ad(f"results/{tissue}/objects/adata_grouped_lineage_refined_clusters.h5ad")

    sc.settings.figdir=f"results/{tissue}/figures/UMAP/annotation"
    sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")
    sc.pl.umap(round1_adata, color="leiden", save="_filtered_reproducible_clusters_all.png")
    

# execute
if __name__ == '__main__':
    main()

# To run CellTypist after
# mkdir -p results/all/figures/annotation
# mkdir -p results/all/tables/annotation/CellTypist
# mkdir -p results/all/figures/UMAP/annotation
# python bin/007-CellTypist.py --tissue "all" --h5_path "results/all/objects/adata_grouped_lineage_refined_clusters.h5ad" --pref_matrix "results_round1/all/tables/batch_correction/best_batch_method.txt" --model "/lustre/scratch127/cellgen/cellgeni/cakirb/celltypist_models/megagut_celltypist_lowerGI+lym_adult_mar24.pkl" --model_name "Megagut_adult_lower_GI_mar24"

    