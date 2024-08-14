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
import nmad_qc

# Decide thresholds and outdir
relative_nMAD_threshold=3
min_ncells_per_sample=10
max_ncells_per_sample=1e5
min_median_nGene_per_samp_blood=0
min_median_nGene_per_samp_gut=0
min_median_nCount_per_samp_blood=0
min_median_nCount_per_samp_gut=0
sample_removal="relative"
outdir="../results"
within="tissue"
cols = ["pct_counts_gene_group__mito_transcript", "log_n_genes_by_counts", "log_total_counts"]
thresholds = {"pct_counts_gene_group__mito_transcript": {"blood": 20, "r": 50, "ti": 50}, "log_n_genes_by_counts": np.log10(250), "log_total_counts": np.log10(500)}
over_under = {"pct_counts_gene_group__mito_transcript": "under", "log_n_genes_by_counts": "over", "log_total_counts": "over"}
threshold_method="outer"
outdir=f"{outdir}/nmad_{relative_nMAD_threshold}-method_{threshold_method}-min_median_nGene_{min_median_nGene_per_samp_gut}-sample_removal_{sample_removal}"
if os.path.exists(outdir) == False:
        os.mkdir(outdir)

# Read in the obs per lineage
lineages = ["Mesenchymal", "Myeloid", "T", "B", "Epithelial"]
linobs = []
for l in lineages:
    print(f"********************* {l} *********************")
    f = f"input_cluster_within_lineage/adata_manual_lineage_clean_all_{l}.h5ad"
    f2 = File(f, 'r')
    # Read only cell-metadata
    obs = read_elem(f2['obs'])
    linobs.append(obs)
    

# Run the QC for each lineage one after another
for index, l in enumerate(lineages):
    print(f"********************* {l} *********************")
    obs = linobs[index]
    # Update outdir
    loutdir = f"{outdir}/{l}"
    if os.path.exists(loutdir) == False:
        os.mkdir(loutdir)
    #
    # Run the filtration like before
    adata = ad.AnnData(obs=obs)
    if l == "Epithelial":
        adata = adata[adata.obs['tissue'] != "blood"]
    #
    # Chuck samples with very few cells
    samp_data = np.unique(adata.obs.samp_tissue, return_counts=True)
    cells_sample = pd.DataFrame({'sample': samp_data[0], 'Ncells':samp_data[1]})
    chuck = cells_sample.loc[cells_sample['Ncells'] < min_ncells_per_sample, 'sample'].values
    if len(chuck) > 0:
        adata = adata[~adata.obs['samp_tissue'].isin(chuck)]
    
    adata.obs['log_n_genes_by_counts'] = np.log10(adata.obs['n_genes_by_counts'])
    adata.obs['log_total_counts'] = np.log10(adata.obs['total_counts'])
    for c in cols:
        nmad_qc.dist_plot(adata, c, within="tissue", relative_threshold=relative_nMAD_threshold, absolute=thresholds[c], out=loutdir)
    #
    for c in cols:
        nmad_qc.update_obs_qc_plot_thresh(adata, c, within="tissue", relative_threshold=relative_nMAD_threshold, threshold_method = threshold_method, relative_directionality = "bi", absolute=thresholds[c], absolute_directionality = over_under[c], plot =True, out=loutdir, out_suffix = "_THRESHOLDED")
    #
    # Filter based on cells
    keep_columns = adata.obs.filter(like='_keep').columns
    adata = adata[adata.obs[keep_columns].all(axis=1)]
    #
    samp_data = np.unique(adata.obs.samp_tissue, return_counts=True)
    depth_count = pd.DataFrame(index = np.unique(adata.obs.samp_tissue), columns=["Mean_nCounts", "nCells", "High_cell_sample", "n_genes_by_counts", "Median_nCounts", "Median_nGene_by_counts", "Median_MT"])
    for s in range(0, depth_count.shape[0]):
        samp = depth_count.index[s]
        depth_count.iloc[s,1] = adata.obs[adata.obs.samp_tissue == samp].shape[0]
        depth_count.iloc[s,0] = adata.obs[adata.obs.samp_tissue == samp].total_counts.sum()/depth_count.iloc[s,1]
        depth_count.iloc[s,3] = adata.obs[adata.obs.samp_tissue == samp].n_genes_by_counts.sum()/depth_count.iloc[s,1]
        depth_count.iloc[s,4] = np.median(adata.obs[adata.obs.samp_tissue == samp].total_counts)
        depth_count.iloc[s,5] = np.median(adata.obs[adata.obs.samp_tissue == samp].n_genes_by_counts)
        depth_count.iloc[s,6] = np.median(adata.obs[adata.obs.samp_tissue == samp].pct_counts_gene_group__mito_transcript)
    #
    adata.obs['disease_tissue'] = adata.obs['tissue'].astype(str) + "_" + adata.obs['disease_status'].astype(str)
    td = np.unique(adata.obs['disease_tissue'])
    for t in td:
        samps = np.unique(adata.obs.loc[adata.obs['disease_tissue'] == t, 'samp_tissue'].values)
        plt.figure(figsize=(8, 6))
        plt.scatter(depth_count[depth_count.index.isin(samps)]["nCells"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c="Navy", alpha=0.7)
        plt.xlabel('nCells')
        plt.ylabel('Median genes detected / cell')
        if t == "blood":
            plt.axvline(x = min_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
            plt.axvline(x = max_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
            plt.axhline(y = min_median_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
            plt.axvline(x = np.median(depth_count[depth_count.index.isin(samps)]["nCells"]), color = 'black', linestyle = '--', alpha = 0.5)
            plt.axhline(y = np.median(depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"]), color = 'black', linestyle = '--', alpha = 0.5)
            plt.title(f"{t} - {min_ncells_per_sample} < nCells < {max_ncells_per_sample}, median_nGene > {min_median_nGene_per_samp_blood}. Black = medians")
        else:
            plt.axvline(x = min_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
            plt.axvline(x = max_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
            plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
            plt.axvline(x = np.median(depth_count[depth_count.index.isin(samps)]["nCells"]), color = 'black', linestyle = '--', alpha = 0.5)
            plt.axhline(y = np.median(depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"]), color = 'black', linestyle = '--', alpha = 0.5)
            plt.title(f"{t} - {min_ncells_per_sample} < nCells < {max_ncells_per_sample}, median_nGene >  {min_median_nGene_per_samp_gut}. Black = medians")
        #
        plt.savefig(f"{loutdir}/sample_ncells_median_ngenes_{t}.png", bbox_inches='tight')
        plt.clf()
    #
    # Filter at sample level
    if "blood" in adata.obs['tissue']:
        blood_keep = (depth_count['Median_nCounts'] > min_median_nCount_per_samp_blood) & (depth_count['Median_nGene_by_counts'] > min_median_nGene_per_samp_blood)
        blood_keep = blood_keep & depth_count.index.isin(adata.obs[adata.obs['tissue'] == "blood"]['samp_tissue'])
        blood_keep = blood_keep[blood_keep == True].index
        gut_keep = (depth_count['Median_nCounts'] > min_median_nCount_per_samp_gut) & (depth_count['Median_nGene_by_counts'] > min_median_nGene_per_samp_gut)
        gut_keep = gut_keep & depth_count.index.isin(adata.obs[adata.obs['tissue'] != "blood"]['samp_tissue'])
        gut_keep = gut_keep[gut_keep == True].index
        both_keep = list(np.concatenate([blood_keep, gut_keep]))
    else:
        gut_keep = (depth_count['Median_nCounts'] > min_median_nCount_per_samp_gut) & (depth_count['Median_nGene_by_counts'] > min_median_nGene_per_samp_gut)
        gut_keep = gut_keep & depth_count.index.isin(adata.obs[adata.obs['tissue'] != "blood"]['samp_tissue'])
        gut_keep = gut_keep[gut_keep == True].index
        both_keep = gut_keep
    #
    #
    if "samples_keep" not in depth_count.columns:
        depth_count['samples_keep'] = depth_count.index.isin(both_keep)
    #
    if sample_removal == "absolute":
        # Apply only if epithelial
        if l == "Epithelial":
            adata = adata[adata.obs['samp_tissue'].isin(depth_count[depth_count['samples_keep']].index)]
    else:
        # Plot the thresholds
        lost_all=[]
        chuck=[]
        for t in td:
            print("--------- Computing relative threshold at a sample level --------")
            samps = np.unique(adata.obs.loc[adata.obs['disease_tissue'] == t, 'samp_tissue'].values)
            data = depth_count.loc[depth_count.index.isin(samps), "Median_nGene_by_counts"].values
            absolute_diff = np.abs(data - np.median(data))
            mad = np.median(absolute_diff)
            cutoff_low = np.median(data) - (relative_nMAD_threshold*mad)
            cutoff_high = np.median(data) + (relative_nMAD_threshold*mad)
            plt.figure(figsize=(8, 6))
            plt.scatter(depth_count[depth_count.index.isin(samps)]["nCells"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c="Navy", alpha=0.7)
            plt.xlabel('nCells')
            plt.ylabel('Median genes detected / cell')
            if t == "blood":
                plt.axvline(x = min_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axvline(x = max_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axhline(y = min_median_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axvline(x = np.median(depth_count[depth_count.index.isin(samps)]["nCells"]), color = 'black', linestyle = '--', alpha = 0.5)
                plt.axhline(y = np.median(depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"]), color = 'black', linestyle = '--', alpha = 0.5)
                plt.title(f"{t} - {min_ncells_per_sample} < nCells < {max_ncells_per_sample}, median_nGene > {min_median_nGene_per_samp_blood}. Black = medians")
            else:
                plt.axvline(x = min_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axvline(x = max_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axvline(x = np.median(depth_count[depth_count.index.isin(samps)]["nCells"]), color = 'black', linestyle = '--', alpha = 0.5)
                plt.axhline(y = np.median(depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"]), color = 'black', linestyle = '--', alpha = 0.5)
                plt.title(f"{t} - {min_ncells_per_sample} < nCells < {max_ncells_per_sample}, median_nGene >  {min_median_nGene_per_samp_gut}. Black = medians")
            #
            plt.axhline(y = cutoff_high, color = 'orange', linestyle = '--', alpha = 0.5)
            plt.axhline(y = cutoff_low, color = 'orange', linestyle = '--', alpha = 0.5)
            plt.savefig(f"{loutdir}/sample_ncells_median_ngenes_{t}_THRESHOLDED.png", bbox_inches='tight')
            plt.clf()
            #
            # print how many samples / cells this would lose
            temp = depth_count[depth_count.index.isin(samps)]
            above = temp.iloc[data > cutoff_high,:]
            below = temp.iloc[data < cutoff_low,:]
            both = pd.concat([above, below])
            sum = pd.DataFrame({"tissue_disease": [t], "n_sample_lost": both.shape[0], "perc_sample_lost": 100*both.shape[0]/temp.shape[0], "n_cell_lost": both['nCells'].sum(), "perc_cell_lost": 100*(both['nCells'].sum()/obs.shape[0])})
            lost_all.append(sum)   
            chuck.append(both)
        #
        print("Samples/cells lost")
        lost=pd.concat(lost_all)
        lost.to_csv(f"{loutdir}/relative_sample_removal_lost_cells.csv")
        # Filter
        chuck_all=pd.concat(chuck)
        adata = adata[~adata.obs['samp_tissue'].isin(chuck_all.index)]
    #
    # Now plot the distribution of cells that have been kept
    postdir = f"{loutdir}/post"
    if os.path.exists(postdir) == False:
            os.mkdir(postdir)
    #
    for c in cols:
        nmad_qc.dist_plot(adata, c, within="tissue", relative_threshold=relative_nMAD_threshold, absolute=thresholds[c], out=postdir)
    #
    # Count cells per tissue
    counts = adata.obs['tissue'].value_counts()
    counts_df = counts.reset_index()
    counts_df.columns = ['tissue', 'count']
    total_sum = counts_df['count'].sum()
    total_row = pd.DataFrame([['Total', total_sum]], columns=['tissue', 'count'])
    counts_df = pd.concat([counts_df, total_row], ignore_index=True)
    counts_df.to_csv(f"{postdir}/total_sum.csv")



###################################
###################################
# Extract the median and mad for each lineage and each column
# HERE: Do this doe lots of different MADs
mads_test = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
mads_per_val = []
for mad_val in mads_test:
    print(f"***************** {mad_val} ************")
    mad_sum=[]
    for index, l in enumerate(lineages):
        print(f"********************* {l} *********************")
        obs = linobs[index]
        adata = ad.AnnData(obs=obs)
        adata.obs['log_n_genes_by_counts'] = np.log10(adata.obs['n_genes_by_counts'])
        adata.obs['log_total_counts'] = np.log10(adata.obs['total_counts'])
        per_c = []
        for c in cols:
            print(f"~~~~~~ {c} ~~~~~~")
            data = adata.obs[c].values
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
plt.savefig(f"../results/mad_range_per_param_per_lineage", bbox_inches='tight')
plt.clf()


# If we pick a MAD of 3 for immune, which value is most comparable to this for epithelium? 
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
dist_imm3_all.to_csv("../results/epi_range_distance_to_imm_range.csv")

#####################################
####################################
# testing defining a relative cut off for samples
lineage="Epithelial"
relative_nMAD_threshold=2
min_median_nCount_per_samp_blood=0
min_median_nGene_per_samp_blood=0
min_median_nCount_per_samp_gut=0
min_median_nGene_per_samp_gut=0
max_ncells_per_sample=1e5
min_ncells_per_sample=10
outdir=f"../results/temp_{lineage}"
if os.path.exists(outdir) == False:
        os.mkdir(outdir)


depth_count = pd.read_csv(f"results/all_{lineage}/tables/depth_count_pre_cell_filtration2.csv")
f = f"results/all_{lineage}/objects/adata_PCAd.h5ad"
f2 = File(f, 'r')
# Read only cell-metadata
obs = read_elem(f2['obs'])
adata = ad.AnnData(obs=obs)
# Within disease x tissue
td = np.unique(adata.obs['tissue_disease'])
depth_count.set_index("samp_tissue", inplace=True)
for t in td:
    samps = np.unique(adata.obs.loc[adata.obs['disease_tissue'] == t, 'samp_tissue'].values)
    plt.figure(figsize=(8, 6))
    plt.scatter(depth_count[depth_count.index.isin(samps)]["nCells"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["High_cell_sample"], alpha=0.7)
    plt.xlabel('nCells')
    plt.ylabel('Median genes detected / cell')
    if t == "blood":
        plt.axvline(x = min_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axvline(x = max_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_median_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axvline(x = np.median(depth_count[depth_count.index.isin(samps)]["nCells"]), color = 'black', linestyle = '--', alpha = 0.5)
        plt.axhline(y = np.median(depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"]), color = 'black', linestyle = '--', alpha = 0.5)
        plt.title(f"{t} - {min_ncells_per_sample} < nCells < {max_ncells_per_sample}, median_nGene > {min_median_nGene_per_samp_blood}. Black = medians")
    else:
        plt.axvline(x = min_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axvline(x = max_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axvline(x = np.median(depth_count[depth_count.index.isin(samps)]["nCells"]), color = 'black', linestyle = '--', alpha = 0.5)
        plt.axhline(y = np.median(depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"]), color = 'black', linestyle = '--', alpha = 0.5)
        plt.title(f"{t} - {min_ncells_per_sample} < nCells < {max_ncells_per_sample}, median_nGene >  {min_median_nGene_per_samp_gut}. Black = medians")
    
    plt.savefig(f"{outdir}/sample_ncells_median_ngenes_{t}.png", bbox_inches='tight')
    plt.clf()
    
    
# Draw lines for relative cut offs: Where would these land? 
lost=[]
for t in td:
    samps = np.unique(adata.obs.loc[adata.obs['disease_tissue'] == t, 'samp_tissue'].values)
    data = depth_count.loc[depth_count.index.isin(samps), "Median_nGene_by_counts"].values
    absolute_diff = np.abs(data - np.median(data))
    mad = np.median(absolute_diff)
    cutoff_low = np.median(data) - (relative_nMAD_threshold*mad)
    cutoff_high = np.median(data) + (relative_nMAD_threshold*mad)
    plt.figure(figsize=(8, 6))
    plt.scatter(depth_count[depth_count.index.isin(samps)]["nCells"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["High_cell_sample"], alpha=0.7)
    plt.xlabel('nCells')
    plt.ylabel('Median genes detected / cell')
    if t == "blood":
        plt.axvline(x = min_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axvline(x = max_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_median_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axvline(x = np.median(depth_count[depth_count.index.isin(samps)]["nCells"]), color = 'black', linestyle = '--', alpha = 0.5)
        plt.axhline(y = np.median(depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"]), color = 'black', linestyle = '--', alpha = 0.5)
        plt.title(f"{t} - {min_ncells_per_sample} < nCells < {max_ncells_per_sample}, median_nGene > {min_median_nGene_per_samp_blood}. Black = medians")
    else:
        plt.axvline(x = min_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axvline(x = max_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
        plt.axvline(x = np.median(depth_count[depth_count.index.isin(samps)]["nCells"]), color = 'black', linestyle = '--', alpha = 0.5)
        plt.axhline(y = np.median(depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"]), color = 'black', linestyle = '--', alpha = 0.5)
        plt.title(f"{t} - {min_ncells_per_sample} < nCells < {max_ncells_per_sample}, median_nGene >  {min_median_nGene_per_samp_gut}. Black = medians")
    
    plt.axhline(y = cutoff_high, color = 'orange', linestyle = '--', alpha = 0.5)
    plt.axhline(y = cutoff_low, color = 'orange', linestyle = '--', alpha = 0.5)
    plt.savefig(f"{outdir}/sample_ncells_median_ngenes_{t}_THRESHOLDED.png", bbox_inches='tight')
    plt.clf()
    
    # print how many samples / cells this would lose
    temp = depth_count[depth_count.index.isin(samps)]
    above = temp.iloc[data > cutoff_high,:]
    below = temp.iloc[data < cutoff_low,:]
    both = pd.concat([above, below])
    sum = pd.DataFrame({"tissue_disease": [t], "n_sample_lost": both.shape[0], "perc_sample_lost": 100*both.shape[0]/temp.shape[0], "n_cell_lost": both['nCells'].sum(), "perc_cell_lost": 100*(both['nCells'].sum()/obs.shape[0])})
    lost.append(sum)
    

lost = pd.concat(lost)