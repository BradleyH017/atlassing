import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



fpath="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/results/all_epithelial/objects/adata_clusters_post_clusterQC.h5ad"
baseout = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/results/all_epithelial/figures/UMAP"
adata = sc.read_h5ad(fpath)
adata.shape

# Calculate the number of cells/sample, in addition to median nGene per cell and nUMI per cell
samp_data = np.unique(adata.obs.samp_tissue, return_counts=True)
cells_sample = pd.DataFrame({'sample': samp_data[0], 'Ncells':samp_data[1]})
high_samps = np.array(cells_sample.loc[cells_sample.Ncells > 10000, "sample"])
low_samps = np.array(cells_sample.loc[cells_sample.Ncells < 500, "sample"])
depth_count = pd.DataFrame(index = np.unique(adata.obs.samp_tissue), columns=["Mean_nCounts", "nCells", "High_cell_sample", "n_genes_by_counts", "Median_nCounts", "Median_nGene_by_counts"])
for s in range(0, depth_count.shape[0]):
    samp = depth_count.index[s]
    depth_count.iloc[s,1] = adata.obs[adata.obs.samp_tissue == samp].shape[0]
    depth_count.iloc[s,0] = sum(adata.obs[adata.obs.samp_tissue == samp].total_counts)/depth_count.iloc[s,1]
    depth_count.iloc[s,3] = sum(adata.obs[adata.obs.samp_tissue == samp].n_genes_by_counts)/depth_count.iloc[s,1]
    depth_count.iloc[s,4] = np.median(adata.obs[adata.obs.samp_tissue == samp].total_counts)
    depth_count.iloc[s,5] = np.median(adata.obs[adata.obs.samp_tissue == samp].n_genes_by_counts)
    if samp in high_samps:
        depth_count.iloc[s,2] = "Red"
    else: 
        depth_count.iloc[s,2] = "Navy"
    # Also annotate the samples with low number of cells - Are these sequenced very deeply?
    if samp in low_samps:
        depth_count.iloc[s,2] = "Green"

depth_count["log10_Mean_Counts"] = np.log10(np.array(depth_count["Mean_nCounts"].values, dtype = "float"))
depth_count["log10_Median_nCounts"] = np.log10(np.array(depth_count["Median_nCounts"].values, dtype = "float"))
depth_count["log10_Median_nGene_by_counts"] = np.log10(np.array(depth_count['Median_nGene_by_counts'].values, dtype="float"))
depth_count["log10_nCells"] = np.log10(np.array(depth_count['nCells'].values, dtype="float"))

# Add nMads of nCells, log10 median nUMI and total_counts
cols = ["nCells", "log10_nCells", "log10_Median_nCounts", "log10_Median_nGene_by_counts"]

def nmad_calc(data):
    absolute_diff = np.abs(data - np.median(data))
    mad = np.median(absolute_diff)
    # Use direction aware filtering
    nmads = (data - np.median(data))/mad
    return(nmads)

def nmad_append(df, var, group=[]):
    # Calculate
    if group:
        vals = df.groupby(group)[var].apply(nmad_calc)
        # Return to df
        temp = pd.DataFrame(vals)
        temp.reset_index(inplace=True)
        if "level_1" in temp.columns:
            temp.set_index("level_1", inplace=True)
        else: 
            temp.set_index("cell", inplace=True)
        
        #temp = temp[var]
        temp = temp.reindex(df.index)
        return(temp[var])
    else:
        vals = nmad_calc(df[var])
        return(vals)
    
# Add nmad of these
for c in cols:
    print(f"nMad_{c}")
    depth_count[f"nMad_{c}"] = nmad_append(depth_count, c)

    
# Add this to anndata
depth_count.reset_index(inplace=True)
depth_count.rename(columns={"index": "samp_tissue"}, inplace=True)
adata.obs.reset_index(inplace=True)
adata.obs = adata.obs.merge(depth_count, how="left", on="samp_tissue")
adata.obs.set_index("cell", inplace=True)

# Plot on UMAP
sc.settings.figdir=baseout
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")

for c in cols:
    print(c)
    sc.pl.umap(adata, color=c, save=f"_{c}_sample_level.png")
    sc.pl.umap(adata, color=f"nMad_{c}", save=f"_{c}_nMads_sample_level.png")

    
# If we were to apply thresholds for these at a level of defined value - How would this look?
nMad_threshold = 2.5
for c in cols:
    print(c)
    adata.obs[f'keep_nMad_{c}'] = np.abs(adata.obs[f"nMad_{c}"]) < nMad_threshold
    nloss=sum(1-adata.obs[f'keep_nMad_{c}'])
    adata.obs[f'keep_nMad_{c}'] = adata.obs[f'keep_nMad_{c}'].astype(str)
    print(f"Would lose {nloss} cells")
    sc.pl.umap(adata, color=f'keep_nMad_{c}', save=f"_{c}_nMads_sample_level_thresholded_2pt5.png")

# Also plot the distribution of each of these metrics
distout="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/results/all_epithelial/figures/QC"
for c in cols:
    print(c)
    data = depth_count[c]
    depth_count[f'keep_nMad_{c}'] = np.abs(depth_count[f"nMad_{c}"]) < nMad_threshold
    absolute_diff = np.abs(data - np.median(data))
    mad = np.median(absolute_diff)
    cutoff_low = np.median(data) - (float(nMad_threshold) * mad)
    cutoff_high = np.median(data) + (float(nMad_threshold) * mad)
    if "log" in c:
        sns.distplot(data, hist=False, rug=True, label=f'(relative): {10**cutoff_low:.2f}-{10**cutoff_high:.2f}')
    else:
        sns.distplot(data, hist=False, rug=True, label=f'(relative): {cutoff_low:.2f}-{cutoff_high:.2f}')
    
    plt.xlabel(f"Sample-level: {c}")
    plt.legend()
    plt.axvline(x = cutoff_low, linestyle = '--', alpha = 0.5)
    plt.axvline(x = cutoff_high, linestyle = '--', alpha = 0.5)
    total_cells = sum(depth_count['nCells'])
    total_samps = depth_count.shape[0]
    subset = depth_count[depth_count[f'keep_nMad_{c}']]
    nkeep_cells = sum(subset['nCells'])
    nkeep_samples = subset.shape[0]
    plt.title(f"Loss of {100*((total_cells - nkeep_cells)/total_cells):.2f}% cells, {100*((total_samps - nkeep_samples)/total_samps):.2f}% samples")
    plt.savefig(f"{distout}/{c}_distribution.png", bbox_inches='tight')
    plt.clf()
