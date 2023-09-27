##### Bradley September 2023
##### Checking the atlassing of the rectum
##### conda activate sc4

# Load in the libraries
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib as mp
from matplotlib import pyplot as plt
from matplotlib.pyplot import rc_context
import kneed as kd
import scvi
import sys
import csv
import datetime
import seaborn as sns
import re
#sys.path.append('/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/conda_envs/scRNAseq/CellRegMap')
#from cellregmap import run_association, run_interaction, estimate_betas
print("Loaded libraries")

# Changedir
import os
cwd = os.getcwd()
print(cwd)
os.chdir("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results")
cwd = os.getcwd()
print(cwd)

# Define variables and options
n=350
min_dist=0.5
spread=0.5
data_name="rectum"
status="healthy"
category="All_not_gt_subset"
param_sweep_path = data_name + "/" + status + "/" + category + "/objects/adata_objs_param_sweep"

# include the 11 high cell samples?
inc_high_cell_samps = False
if inc_high_cell_samps == True:
    nn_file=param_sweep_path + "/inc_11_high_cell_samps/NN_{}_scanvi.adata".format(n)
else:
    nn_file=param_sweep_path + "/NN_{}_scanvi.adata".format(n)

# Load NN file
adata = ad.read_h5ad(nn_file)

# Recompute UMAP with percieved optimum conditions
sc.tl.umap(adata, min_dist=min_dist, spread=spread, neighbors_key ="scVI_nn")
# Save post UMAP
umap_file = re.sub("_scanvi.adata", "_scanvi_umap.adata", nn_file)
adata.write_h5ad(umap_file)

# Define fig out dir
sc.settings.figdir=data_name + "/" + status + "/" + category + "/figures"

# Plot categories, labels, samples
sc.pl.umap(adata, color="category",frameon=True, save="_post_batch_post_sweep_category.png")
sc.pl.umap(adata, color="label",frameon=True, save="_post_batch_post_sweep_keras.png")
sc.pl.umap(adata, color="bead_lot",frameon=True, save="_post_batch_post_sweep_bead_lot.png")
sc.pl.umap(adata, color="gem_lot",frameon=True, save="_post_batch_post_sweep_gem_lot.png")
sc.pl.umap(adata, color="convoluted_samplename", frameon=True, save="_post_batch_post_sweep_samples.png", palette=list(mp.colors.CSS4_COLORS.values()))
adata.obs['Keras:predicted_celltype_probability'] = adata.obs['Keras:predicted_celltype_probability'].astype("float")
sc.pl.umap(adata, color="Keras:predicted_celltype_probability", frameon=True, save="_post_batch_post_sweep_keras_conf.png")
# Plot RP%
sc.pl.umap(adata, color="pct_counts_gene_group__ribo_protein", frameon=True, save="_post_batch_post_sweep_rpperc.png")
# Calculate IG% and plot
adata.var[adata.var.gene_symbols.str.contains("IG[HKL]V|IG[KL]J|IG[KL]C|IGH[ADEGM]")]
totsum = adata.layers['counts'].sum(axis=1)
is_ig = adata.var.gene_symbols.str.contains("IG[HKL]V|IG[KL]J|IG[KL]C|IGH[ADEGM]")
temp_counts = adata.layers['counts']
igcounts = temp_counts[:,is_ig.values]
igsum = igcounts.sum(axis=1)
ig_perc = np.array(igsum)/np.array(totsum)
adata.obs['ig_perc'] = ig_perc
sc.pl.umap(adata, color="ig_perc", frameon=True, save="_post_batch_post_sweep_igperc.png")


# Still looks like one samples isn't integrating very well
# Plot with axis and find this one
umap = pd.DataFrame(adata.obsm['X_umap'])
umap.columns = ["UMAP_1", "UMAP_2"]
umap['Sample'] = adata.obs.convoluted_samplename.values
plt.figure(figsize=(8, 6))
scatter = plt.scatter(umap['UMAP_1'], umap['UMAP_2'], s=0.1)
x_ticks = np.arange(min(umap['UMAP_1']),max(umap['UMAP_1']), 0.3)
plt.xticks(x_ticks)
y_ticks = np.arange(min(umap['UMAP_2']),max(umap['UMAP_2']), 0.3)
plt.yticks(y_ticks)
plt.xlabel('UMAP_1')
plt.ylabel('UMAP_2')
plt.tick_params(axis='both', labelsize=3)
plt.savefig(data_name + "/" + status + "/" + category + "/figures/umap_post_batch_post_sweep_sample_ticks.png", dpi=300, bbox_inches='tight')
plt.clf()

# Plot this non-integrating sample
umap_x_min = 4.35
umap_x_max = 5.25
umap_y_min = 6.38
umap_y_max = 7.28
# Create a boolean mask based on the UMAP coordinates
umap_mask = (
    (adata.obsm['X_umap'][:, 0] >= umap_x_min) & (adata.obsm['X_umap'][:, 0] <= umap_x_max) &
    (adata.obsm['X_umap'][:, 1] >= umap_y_min) & (adata.obsm['X_umap'][:, 1] <= umap_y_max)
)
# Subset the AnnData object based on the mask
subset_adata = adata[umap_mask]
# Plot, coloured by sample
sc.pl.umap(subset_adata, color="convoluted_samplename", frameon=True, save="_post_batch_post_sweep_subset_coords1_samples.png", palette = 'Set3')
candidates = ["OTARscRNA13781777", "OTARscRNA13781786"]
subset_adata.obs['candidate_bad'] = adata.obs['convoluted_samplename'].apply(lambda x: x if x in candidates else 'No')
# Plot these ones only 
sc.pl.umap(subset_adata, color="candidate_bad", frameon=True, save="_post_batch_post_sweep_subset_coords1_samples_cols.png")
problem_sample = "OTARscRNA13781777"
# Plot UMAP of all samples, with this one highlighted
adata.obs['prob_bad'] = adata.obs['convoluted_samplename'].apply(lambda x: x if x in problem_sample else 'No')
sc.pl.umap(adata, color="c", frameon=True, save="_post_batch_post_sweep_bad.png")


# Looks like this sample has very few cells, and mostly immune - check this
np.unique(adata.obs.prob_bad, return_counts=True)
cats = np.unique(adata.obs.category)
count_cat = pd.DataFrame(np.unique(adata.obs[adata.obs.convoluted_samplename == problem_sample].category, return_counts=True)).T
count_cat.columns = ['label', 'freq']
plt.figure(figsize=(8, 6))
plt.pie(count_cat.freq, labels = count_cat.label, autopct='%.0f%%')
plt.savefig(data_name + "/" + status + "/" + category + "/figures/umap_post_batch_post_sweep_bad_samples_prop.png", dpi=300, bbox_inches='tight')

# Check a random for categories
random_samps = np.random.choice(np.unique(adata.obs.convoluted_samplename), 3)
for index,s in enumerate(random_samps):
    count_cat = pd.DataFrame(np.unique(adata.obs[adata.obs.convoluted_samplename == s].category, return_counts=True)).T
    count_cat.columns = ['label', 'freq']
    plt.figure(figsize=(8, 6))
    plt.pie(count_cat.freq, labels = count_cat.label, autopct='%.0f%%')
    plt.savefig(data_name + "/" + status + "/" + category + "/figures/umap_post_batch_post_sweep_random_good_samples_prop" + str(index) + ".png", dpi=300, bbox_inches='tight')



# Plot within category
cats = np.unique(adata.obs.category)
for c in cats:
    print(c)
    temp = adata[adata.obs.category == c]
    sc.pl.umap(temp, color="label", title=c, frameon=False , save="_" + c + "_post_batch_post_sweep_keras.png")


# Also plot ngenes, n reads
sc.pl.umap(adata, color="ngenes_per_cell", frameon=True, save="_post_batch_post_sweep_ngenes.png")
# And total counts
sc.pl.umap(adata, color="total_counts", frameon=True, save="_post_batch_post_sweep_ncounts.png")
# And MT%
sc.pl.umap(adata, color="pct_counts_gene_group__mito_transcript", frameon=True, save="_post_batch_post_sweep_mtperc.png")

# Check categories, coloured by MT%
cats = np.unique(adata.obs.category)
def mad(data):
    return np.median(np.abs(data - np.median(data)))

for c in cats:
    print(c)
    temp = adata[adata.obs.category == c]
    sc.pl.umap(temp, color="pct_counts_gene_group__mito_transcript", title=c, frameon=False , save="_" + c + "_post_batch_post_sweep_mtperc.png")
    # Print the median + 2.5*the median absolute deviation for each celltype
    #mt_perc=temp.obs["pct_counts_gene_group__mito_transcript"]
    #median_plus_2pt5_mad = np.median(mt_perc) + 2.5*(mad(mt_perc))
    #print("For {}, the median={} and the median+2.5*MAD={}".format(c, np.median(mt_perc), median_plus_2pt5_mad))


# Is there a significant proportion of MT%, RP% and HLA% in the HVGs?
hvg = adata.var[adata.var.highly_variable == True]
mt_perc = hvg[hvg['gene_symbols'].str.contains("MT-")].shape[0]/hvg.shape[0]

# If we grouped cells into their lineages, what would relative cut off for MT perc be?
lin_df = pd.DataFrame({'category': cats, 'lineage': ""}) 
lin_df['lineage'] = np.where(lin_df['category'].isin(['B Cell', 'B Cell plasma', 'T Cell', 'Myeloid']), 'Immune', lin_df['lineage'])
lin_df['lineage'] = np.where(lin_df['category'].isin(['Stem cells', 'Secretory', 'Enterocyte']), 'Epithelial', lin_df['lineage'])
lin_df['lineage'] = np.where(lin_df['category']== 'Mesenchymal', 'Mesenchymal', lin_df['lineage'])
# Merge onto adata
adata.obs.index = adata.obs.index.astype(str)
cells = adata.obs.index
adata.obs = adata.obs.merge(lin_df, on='category', how="left")
adata.obs.index = cells

# Plot this on umap
sc.pl.umap(adata, color="lineage",frameon=True, save="_post_batch_post_sweep_lineage.png")


# Plot the MT% per category, adding a line for the relative cut off
for index,c in enumerate(cats):
    print(c)
    temp = adata[adata.obs.category == c]
    mt_perc=temp.obs["pct_counts_gene_group__mito_transcript"]
    median_plus_2pt5_mad = np.median(mt_perc) + 2.5*(mad(mt_perc))
    print(median_plus_2pt5_mad)
    if index == 0:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
    sns.distplot(mt_perc, hist=False, rug=True, label=c)
    if median_plus_2pt5_mad < 50:
        plt.axvline(median_plus_2pt5_mad, color=sns.color_palette()[cats.tolist().index(c)], linestyle='--')
    else:
        plt.axvline(50, color=sns.color_palette()[cats.tolist().index(c)], linestyle='--')


plt.legend()
plt.xlabel('MT%')
ax.set(xlim=(0, 50))
plt.savefig(data_name + "/" + status + "/" + category + "/figures" + '/postQC_relative_mt_perc_per_cat.png', bbox_inches='tight')
plt.clf()

# Now do this per lineage
lins = np.unique(lin_df.lineage)
for index,c in enumerate(lins):
    print(c)
    temp = adata[adata.obs.lineage == c]
    mt_perc=temp.obs["pct_counts_gene_group__mito_transcript"]
    median_plus_2pt5_mad = np.median(mt_perc) + 2.5*(mad(mt_perc))
    print(median_plus_2pt5_mad)
    if index == 0:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
    sns.distplot(mt_perc, hist=False, rug=True, label=c)
    if median_plus_2pt5_mad < 50:
        plt.axvline(median_plus_2pt5_mad, color=sns.color_palette()[lins.tolist().index(c)], linestyle='--')
    else:
        plt.axvline(50, color=sns.color_palette()[cats.tolist().index(c)], linestyle='--')
    nlost = len(mt_perc[mt_perc > median_plus_2pt5_mad])
    print("The number of cells lost in the {} lineage if we use this cut off would be {}".format(c, nlost))


plt.legend()
plt.xlabel('MT%')
ax.set(xlim=(0, 50))
plt.savefig(data_name + "/" + status + "/" + category + "/figures" + '/postQC_relative_mt_perc_per_lineage.png', bbox_inches='tight')
plt.clf()


# It is possible that the mixing is occuring between populations that are of low confidence (compared with the TI)
# Plot this divided by category
for c in cats:
    print(c)
    temp = adata[adata.obs.category == c]
    sc.pl.umap(temp, color="Keras:predicted_celltype_probability", title=c, frameon=False , save="_" + c + "_post_batch_post_sweep_keras_conf.png")





# Have a look at the ngenes / cell / sample
samp_data = np.unique(adata.obs.convoluted_samplename, return_counts=True)
cells_sample = pd.DataFrame({'sample': samp_data[0], 'Ncells':samp_data[1]})
high_samps = np.array(cells_sample.loc[cells_sample.Ncells > 10000, "sample"])
bad_samps = problem_sample
depth_count = pd.DataFrame(index = np.unique(adata.obs.convoluted_samplename), columns=["Mean_nCounts", "nCells", "High_cell_sample", "ngenes_per_cell", "sample"])
for s in range(0, depth_count.shape[0]):
    samp = depth_count.index[s]
    depth_count.iloc[s,1] = adata.obs[adata.obs.convoluted_samplename == samp].shape[0]
    depth_count.iloc[s,0] = sum(adata.obs[adata.obs.convoluted_samplename == samp].total_counts)/depth_count.iloc[s,1]
    depth_count.iloc[s,3] = sum(adata.obs[adata.obs.convoluted_samplename == samp].ngenes_per_cell)/depth_count.iloc[s,1]
    if samp in high_samps:
        depth_count.iloc[s,2] = "Red"
        depth_count.iloc[s,4] = samp
    else: 
         depth_count.iloc[s,2] = "Navy"
    # Also annotate the samples with low number of cells - Are these sequenced very deeply?
    if samp in bad_samps:
        depth_count.iloc[s,2] = "Green"
        depth_count.iloc[s,4] = samp

depth_count["log10_Mean_Counts"] = np.log10(np.array(depth_count["Mean_nCounts"].values, dtype = "float"))
# Plot again
plt.figure(figsize=(8, 6))
plt.scatter(depth_count["Mean_nCounts"], depth_count["ngenes_per_cell"],  c=depth_count["High_cell_sample"], alpha=0.7)
plt.xlabel('Mean counts / cell')
plt.ylabel('Mean genes detected / cell')
plt.savefig(data_name + "/" + status + "/" + category + "/figures/post_batch_post_sweep_counts_cells_genes_cells_labels.png", bbox_inches='tight')
plt.clf()

# Look at the distribution of log10genes detected for bad samples and all samples
# Also save max density onto the table to look at
depth_count[["max_density"]] = 0
samps = np.unique(cells_sample["sample"])
for index,s in enumerate(samps):
    data = adata.obs[adata.obs.convoluted_samplename == s].ngenes_per_cell
    if index == 0:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
    if s in np.hstack((np.array(bad_samps, dtype="object"), bad_samps)):
        sns.distplot(data, hist=False, rug=True, label=s, kde_kws={'linewidth': 2.5})
    else:
        sns.distplot(data, hist=False, rug=True, color='black')
    # Find the max value (remove this code if just want to plot the problem ones)
    kde = sns.kdeplot(data, shade=True)
    depth_count.iloc[index,6] = kde.get_lines()[index].get_ydata().max()

plt.legend()
plt.xlabel('nGenes detected/cell')
ax.set(xlim=(0, max(adata.obs.ngenes_per_cell)))
# Set y-axis limits to match the maximum density
ax.set_ylim(0, 0.0017)
plt.savefig(data_name + "/" + status + "/" + category + "/figures/post_batch_genes_per_sample.png", bbox_inches='tight')
plt.clf()

# Have a look at the ones with very high low ngenes/cell
# cut off at 0.0008 (initially, removed the very high cell number, bad quality samples)
# Now looking like 0.0006 may capture additional, badly integrating samples
cutoff=0.0006
potbadsamps = depth_count[depth_count.max_density > cutoff]
potbadsamps = np.array(potbadsamps.index)
for index,s in enumerate(samps):
    data = adata.obs[adata.obs.convoluted_samplename == s].ngenes_per_cell
    if index == 0:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
    if s in potbadsamps:
        sns.distplot(data, hist=False, rug=True, label=s, kde_kws={'linewidth': 2.5})
    else:
        sns.distplot(data, hist=False, rug=True, color='black')

plt.legend()
plt.xlabel('nGenes detected/cell')
ax.set(xlim=(0, max(adata.obs.ngenes_per_cell)))
# Set y-axis limits to match the maximum density
ax.set_ylim(0, 0.0017)
plt.axhline(y = cutoff, color = 'red', linestyle = '--', alpha = 0.5)
plt.savefig(data_name + "/" + status + "/" + category + "/figures/post_batch_genes_per_potential_bad_sample.png", bbox_inches='tight')
plt.clf()

# Plot these on UMAP
adata.obs['potential_bad_sample'] = adata.obs['convoluted_samplename'].apply(lambda x: x if x in potbadsamps else 'No')
sc.pl.umap(adata, color="potential_bad_sample",frameon=True, save="_post_batch_post_sweep_low_n_gene_cell_samples.png")

# Check the ncells/sample, some of these look like they have very few cells
depth_count[depth_count.index.isin(potbadsamps)]

# Plot the other potential bad samples in the Mean genes detect/cell
for s in range(0, depth_count.shape[0]):
    samp = depth_count.index[s]
    if samp in high_samps:
        depth_count.iloc[s,2] = "Red"
        depth_count.iloc[s,4] = samp
    else: 
         depth_count.iloc[s,2] = "Navy"
    # Also annotate the samples with low number of cells - Are these sequenced very deeply?
    if samp in potbadsamps:
        depth_count.iloc[s,2] = "Green"
        depth_count.iloc[s,4] = samp

# Plot again
plt.figure(figsize=(8, 6))
plt.scatter(depth_count["Mean_nCounts"], depth_count["ngenes_per_cell"],  c=depth_count["High_cell_sample"], alpha=0.7)
plt.xlabel('Mean counts / cell')
plt.ylabel('Mean genes detected / cell')
plt.savefig(data_name + "/" + status + "/" + category + "/figures/post_batch_post_sweep_counts_cells_genes_cells_labels_potbad.png", bbox_inches='tight')
plt.clf()


# Determining new cut off
depth_count[["cross_density"]] = 0
for index,s in enumerate(samps):
    data = adata.obs[adata.obs.convoluted_samplename == s].ngenes_per_cell
    if index == 0:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
    if s in potbadsamps:
        sns.distplot(data, hist=False, rug=True, label=s, kde_kws={'linewidth': 2.5})
    else:
        sns.distplot(data, hist=False, rug=True, color='black')
    # Find the max value (remove this code if just want to plot the problem ones)
    kde = sns.kdeplot(data, shade=False)
    ax = plt.gca()
    x_values, y_values = ax.lines[index].get_data()
    # Get x when y crosses
    x_thresh = x_values[y_values >= cutoff]
    # Take the last of these
    if len(x_thresh > 0):
        threshold_x = x_thresh[len(x_thresh)-1]
        print(threshold_x)
        depth_count.iloc[index,7] = threshold_x

plt.legend()
plt.xlabel('nGenes detected/cell')
ax.set(xlim=(0, max(adata.obs.ngenes_per_cell)))
# Set y-axis limits to match the maximum density
ax.set_ylim(0, 0.0017)
plt.axhline(y = cutoff, color = 'red', linestyle = '--', alpha = 0.5)
plt.axvline(x = max(depth_count.cross_density), color = 'red', linestyle = '--', alpha = 0.5)
plt.savefig(data_name + "/" + status + "/" + category + "/figures/post_batch_genes_per_potential_bad_sample_newthresh.png", bbox_inches='tight')
plt.clf()
print(max(depth_count.cross_density))
depth_count.cross_density[depth_count.cross_density > 0]


# Define cells as above or below a cut off and plot
ngene_cutoff = 300
adata.obs['low_ngene_cell'] = adata.obs['ngenes_per_cell'].apply(lambda x: 'Yes' if x < ngene_cutoff else 'No')
sc.pl.umap(adata, color="low_ngene_cell",frameon=True, save="_post_batch_post_sweep_low_n_gene_cell_low_ngene_cell.png")

# Save a list of the potentially bad samples - These will be removed in the next round of integration and analysis
np.savetxt(data_name + "/" + status + "/" + category + '/bad_samples_to_remove.txt', potbadsamps, delimiter='\n', fmt='%s')


# Save umap version
#umap_file = 

# Plot umap with dimensions
umap = pd.DataFrame(adata.obsm['X_umap'])
umap.columns = ["UMAP_1", "UMAP_2"]
umap['Sample'] = adata.obs.convoluted_samplename.values
plt.figure(figsize=(8, 6))
scatter = plt.scatter(umap['UMAP_1'], umap['UMAP_2'], s=0.1, c=np.arange(len(umap['Sample'])), cmap='Set1')
x_ticks = np.arange(min(umap['UMAP_1']),max(umap['UMAP_1']), 0.3)
plt.xticks(x_ticks)
y_ticks = np.arange(min(umap['UMAP_2']),max(umap['UMAP_2']), 0.3)
plt.yticks(y_ticks)
plt.xlabel('UMAP_1')
plt.ylabel('UMAP_2')
plt.tick_params(axis='both', labelsize=3)
plt.colorbar(scatter, label='Sample')
plt.savefig(data_name + "/" + status + "/" + category + "/figures/umap_post_batch_post_sweep_sample_ticks.png", dpi=300, bbox_inches='tight')
plt.clf()


# Plot within category
cats = np.unique(adata.obs.category)
for c in cats:
    print(c)
    temp = adata[adata.obs.category == c]
    sc.pl.umap(temp, color="label", title=c,frameon=False, figsize=(6,4), save="_" + c + "_post_batch_post_sweep_keras.png")

# Rename convoluted sample name (this seems to have been lost after the batch correction). 
# Have checked the adata used for batch correction and this is correct though. So not to worry.
#adata.obs["convoluted_samplename"] = adata.obs['convoluted_samplename'].str.replace("__donor", '')

# Check where the high samples look here
samp_data = np.unique(adata.obs.convoluted_samplename, return_counts=True)
cells_sample = pd.DataFrame({'sample': samp_data[0], 'Ncells':samp_data[1]})
high_cell_samps = np.array(cells_sample.loc[cells_sample.Ncells > 10000, "sample"])
def check_intersection(value):
    if value in high_cell_samps:
        return 'yes'
    else:
        return 'no'

# Create a new column 'intersects' using the apply method
adata.obs['high_cell_samp'] = adata.obs['convoluted_samplename'].apply(lambda x: check_intersection(x))

# Plot the high cell samples labelled by their ID
def fill_column(row):
    if row['high_cell_samp'] == 'yes':
        return row['convoluted_samplename']
    else:
        return 'Not high sample'  # You can also specify a different default value if needed

# Create a new column 'NewColumn' based on the conditions
adata.obs['high_cell_samp_label'] = adata.obs.apply(fill_column, axis=1)
# Plot these
sc.pl.umap(adata, color="high_cell_samp_label", frameon=True, save="_post_batch_post_sweep_high_samples.png")

# Subset adata for problem region to try and identify the non-integrating samples
umap_x_min = 0.4
umap_x_max = 1
umap_y_min = 3.48
umap_y_max = 3.7
# Create a boolean mask based on the UMAP coordinates
umap_mask = (
    (adata.obsm['X_umap'][:, 0] >= umap_x_min) & (adata.obsm['X_umap'][:, 0] <= umap_x_max) &
    (adata.obsm['X_umap'][:, 1] >= umap_y_min) & (adata.obsm['X_umap'][:, 1] <= umap_y_max)
)
# Subset the AnnData object based on the mask
subset_adata = adata[umap_mask]
# Plot, coloured by sample
sc.pl.umap(subset_adata, color="convoluted_samplename", frameon=True, save="_post_batch_post_sweep_subset_coords1_samples.png")

# Have a look at the potentially bad samples (inc. high cell number samples)



# Have a look at the high cell samples seperately and seperately within a couple categories
high = adata[adata.obs.high_cell_samp == "yes"]
sc.pl.umap(high, color="category",title="High cell samples", frameon=False, save="_post_batch_post_sweep_high_cell_number.png")
good = adata[adata.obs.high_cell_samp == "no"]
sc.pl.umap(good, color="category",title="Not high cell samples", frameon=False, save="_post_batch_post_sweep_not_high_cell_number.png")
# For just the B Cells / T Cells
for i, val in enumerate(["B Cell", "T Cell"]):
    temp = adata[adata.obs.category == val]
    high = temp[temp.obs.high_cell_samp == "yes"]
    good = temp[temp.obs.high_cell_samp == "no"]
    sc.pl.umap(high, color="label",title="High cell samples - " + val, frameon=False, save="_" + val  + "_post_batch_post_sweep_high_cell_number.png")
    sc.pl.umap(good, color="label",title="Not high cell samples - " + val, frameon=False, save="_" + val  + "_post_batch_post_sweep_not_high_cell_number.png")






# Assessment of batch effect integration
df = pd.read_csv(data_name + "/" + status + "/" + category + "/tables/integration_benchmarking.csv")



