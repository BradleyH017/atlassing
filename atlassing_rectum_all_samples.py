##################################################
#### Bradley August 2023
# Atlassing the rectum scRNAseq
# Using all of the expression data, not just limited to that which is genotyped.
##################################################

# Change dir
import os
cwd = os.getcwd()
print(cwd)
os.chdir("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results")
cwd = os.getcwd()
print(cwd)

# Load packages
import sys
print(sys.path)
import numpy as np
print(np.__file__)
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib as mp
from matplotlib import pyplot as plt
from matplotlib.pyplot import rc_context
import kneed as kd
import scvi
import csv
import datetime
import seaborn as sns
from formulae import design_matrices
from limix.qc import quantile_gaussianize
import matplotlib.pyplot as plt
import math
import scipy.stats as st
import re
print("Loaded libraries")

# Define the datasets - only running rectum in this script, so don't need to worry about the datset name, disease status or category (am using all here)
data = "../proc_data/2023_09_rectum/adata.h5ad"
status="healthy"
category="All_not_gt_subset"

# Define global figure directory
statpath = "rectum/" + status
catpath = statpath + "/" + category
figpath = catpath + "/figures"
tabdir = catpath + "/tables"
objpath = catpath + "/objects"
if os.path.exists(figpath) == False:
    if os.path.exists(catpath) == False:
        if os.path.exists(statpath) == False:
            os.mkdir(statpath)
        os.mkdir(catpath)
    os.mkdir(figpath)

if os.path.exists(tabdir) == False:
    os.mkdir(tabdir)

if os.path.exists(objpath) == False:
    os.mkdir(objpath)

# Define global figure directory
sc.settings.figdir=figpath
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")

# Load the data
adata = ad.read_h5ad(data)
print("Loaded the data")
# Rename the genotyping ID column so doesn't cause issues later
adata.obs = adata.obs.rename({'Post_QC_Genotyping_ID_(Tobi_Edit)': 'Corrected_Genotyping_ID'}, axis='columns')

# Checking for intersection of pilot samples
pilot = ["OTARscRNA13669848","OTARscRNA13214581","OTARscRNA13236940","OTARscRNA13265818","OTARscRNA13430429","OTARscRNA13430438","OTARscRNA13669841"]
have = np.unique(adata.obs.convoluted_samplename)
already = np.intersect1d(pilot, have)
missing = np.setdiff1d(pilot, have)

# Before analaysis, remove the bad samples
# These were found to not integrate well using scVI
# These also had a majority of cells with low number of genes detected per cell. Raising this threshold made it difficult to determine where a line could be set to remove these non-integrating cells, and immune cells.
# So instead, decision was made to remove the cells from these samples entirely
bad_samps = np.loadtxt(catpath + "/bad_samples_to_remove.txt", dtype=str)
# After integration with just this set, had some additional samples that were not integrating well. 
# These were also those with low ngenes/cell detected. Density = 0.0006
# Repeating this, removing these samples 
additional_bad_samps = np.array(['OTARscRNA10652911', 'OTARscRNA13669850', 'OTARscRNA13781777','OTARscRNA9997698'])
adata = adata[~(adata.obs.convoluted_samplename.isin(bad_samps))]
adata = adata[~(adata.obs.convoluted_samplename.isin(additional_bad_samps))]

####################################
######### Cell filtration ##########
####################################

# First add category for plots later
coarse_fine_label = pd.read_csv("../cluster_annotations/coarse_to_fine.csv")
coarse_fine_label = coarse_fine_label[["label", "category"]]
adata.obs.rename(columns={"Keras:predicted_celltype": "label__machine"}, inplace=True)
label_labelmachine = pd.read_csv("../cluster_annotations/rectum_adata_label_label_machine.csv", index_col=0)
label_labelmachine = label_labelmachine.merge(coarse_fine_label)
cells = adata.obs.index
adata.obs = adata.obs.merge(label_labelmachine, how="left")
adata.obs.index = cells

# Subset for the desired samples only. If healthy, this will mean that 3 samples are lost (2 UC ["5892STDY8966203", "5892STDY8966204"] and one CD [OTARscRNA9294505])
adata = adata[adata.obs.disease_status == status]

# Now do some preliminary preprocessing
# 1. What is the minimum UMI?
print(min(adata.obs[["total_counts"]].values))
import statistics
print(statistics.median(adata.obs[["total_counts"]].values))
# Cut off at 200 
tot_count = adata.obs["total_counts"]
good_tot_count = tot_count > 200
# Plot this
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts'],
             jitter=0.4, multi_panel=True, save=True)
adata.obs["tot_count_gt_200"] = good_tot_count

# Plot per cell category
cats = np.unique(adata.obs.category)
adata.obs['total_counts'] = adata.obs['total_counts'].astype(int)
for c in cats:
    data = np.log10(adata.obs[adata.obs.category == c].total_counts)
    if c == "B Cell":
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
    sns.distplot(data, hist=False, rug=True, label=c)

plt.legend()
plt.xlabel('log10(nUMI)')
plt.axvline(x = np.log10(200), color = 'red', linestyle = '--', alpha = 0.5)
plt.savefig(figpath + '/nUMI_per_category.png', bbox_inches='tight')
plt.clf()

# Subset this
adata = adata[adata.obs.tot_count_gt_200 == True]

# 2. N gene expressed
print("The number of cells initially included is", adata.n_obs)
counts=adata.X
ncounts_gt1=(counts >= 1).sum(axis=1)
express_gt100 = ncounts_gt1 > 100
print("The number of cells with >1 count in ", np.count_nonzero(express_gt100))

# Plot
# All
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
sns.distplot(np.log10(ncounts_gt1))
plt.xlabel('log10(ngenes)')
plt.axvline(x = np.log10(100), color = 'red', linestyle = '--', alpha = 0.5)
plt.savefig(figpath + '/ngenes_all.png', bbox_inches='tight')
plt.clf()
# Per category
for c in cats:
    idx = np.where(adata.obs.category == c)
    data = np.log10(ncounts_gt1[idx])
    if c == "B Cell":
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
    sns.distplot(data, hist=False, rug=True, label=c)

plt.legend()
plt.xlabel('log10(ngenes)')
plt.axvline(x = np.log10(100), color = 'red', linestyle = '--', alpha = 0.5)
plt.savefig(figpath + '/ngenes_category.png', bbox_inches='tight')
plt.clf()

# Subset for this
adata.obs["express_gt100"] = express_gt100
adata = adata[adata.obs.express_gt100 == True]

# 3. MT%
print(adata.obs["pct_counts_gene_group__mito_transcript"].max()) # Not filtered
# Plot this
sc.pl.violin(adata, ['pct_counts_gene_group__mito_transcript'],
             jitter=0.4, multi_panel=True, save="_mt_perc.png")

# Calculate the proportion of lost cells at each cut off
mt_cuts = pd.DataFrame({"cutoff": range(0,110,10), "ncells":0 })
for c in mt_cuts['cutoff'].values:
    mt_cuts.loc[mt_cuts.cutoff == c,'ncells'] = adata.obs[adata.obs.pct_counts_gene_group__mito_transcript < c].shape[0]

mt_cuts = mt_cuts.T
mt_cuts.to_csv(tabdir + "/MT_perc_cutoffs.csv")

## Plot this
mt_cuts = mt_cuts.T
plt.plot(100-mt_cuts.cutoff, mt_cuts.ncells)
plt.title('Loss of cells based on MT%')
plt.xlabel('100 - MT% cut off')
plt.ylabel('NCells retained')
plt.axvline(x = 50, color = 'red', linestyle = '--', alpha = 0.5)
plt.savefig(figpath + '/mt_perc_cutoffs.png', bbox_inches='tight')
plt.clf()

# Define relative cut off (median absolute deviation)
def mad(data):
    return np.median(np.abs(data - np.median(data)))

mt_perc=adata.obs["pct_counts_gene_group__mito_transcript"]
print("The MAD of the remaining cells is:{}".format(mad(mt_perc)))
median_plus_2pt5_mad = np.median(mt_perc) + 2.5*(mad(mt_perc))
nretained = sum(mt_perc < median_plus_2pt5_mad)
print("Using the median + 2.5*mad={}, leaving us with {} cells".format(median_plus_2pt5_mad, nretained))
# Plot this over the line graph also
plt.plot(100-mt_cuts.cutoff, mt_cuts.ncells)
plt.title('Loss of cells based on MT%')
plt.xlabel('100 - MT% cut off')
plt.ylabel('NCells retained')
plt.axvline(x = 50, color = 'red', linestyle = '--', alpha = 0.5)
plt.axvline(x = median_plus_2pt5_mad, color = 'green', linestyle = '--', alpha = 0.5, label="median+2.5*mad")
plt.savefig(figpath + '/mt_perc_cutoffs.png', bbox_inches='tight')
plt.clf()

# Find ranges per category
# Then produce summary statistics per category (using 50% cut off)
mt_per_cat = pd.DataFrame({"category": cats, "95L": 0, "mean": 0, "95U": 0})
for c in cats:
    data = adata.obs[adata.obs.category == c].pct_counts_gene_group__mito_transcript
    lims = st.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    mean = sum(data)/len(data)
    mt_per_cat.loc[mt_per_cat.category == c, "95L"] = lims[0]
    mt_per_cat.loc[mt_per_cat.category == c, "mean"] = mean
    mt_per_cat.loc[mt_per_cat.category == c, "95U"] = lims[1]
    if c == "B Cell":
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
    sns.distplot(data, hist=False, rug=True, label=c)

plt.legend()
plt.xlabel('MT%')
plt.axvline(x = 50, color = 'red', linestyle = '--', alpha = 0.5)
ax.set(xlim=(0, 100))
plt.savefig(figpath + '/mt_perc_50pct_per_category.png', bbox_inches='tight')
plt.clf()

# How many B Cell Plasma cells are lost with a cut off of 50%
retained_bcp = adata.obs[(adata.obs.category == "B Cell plasma") & (adata.obs.pct_counts_gene_group__mito_transcript < 50)].shape[0]
total_bcp = adata.obs[(adata.obs.category == "B Cell plasma")].shape[0]
print("The number of lost B Cell plasma cells lost at a cut off of 50 MT% is {}({}%)".format(total_bcp-retained_bcp, 100*(total_bcp-retained_bcp)/total_bcp))

# Add to adata and cut off
cutoff=50
mt_perc=adata.obs["pct_counts_gene_group__mito_transcript"]
mt_perc_ls50 = mt_perc < cutoff
adata.obs["mt_perc_less_50"] = mt_perc_ls50
adata=adata[adata.obs.mt_perc_less_50 == True] #325,318 cells
print("The number of cells that remain after filtration is", adata.n_obs)

# 4. Probabilityof doublet. This has already been filtered by the yascp pipeline
print(adata.obs["prob_doublet"].max()) # 0.29

# 5. Label machine. What is the cut off for confidence ( don't have this in the rectum, also don't want to subset for this at this inital point. )
#


# Looking at the number of retained cells per patient
samp_data = np.unique(adata.obs.convoluted_samplename, return_counts=True)
cells_sample = pd.DataFrame({'sample': samp_data[0], 'Ncells':samp_data[1]})
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
sns.distplot(cells_sample.Ncells)
plt.xlabel('Cells/sample')
plt.axvline(x = 500, color = 'red', linestyle = '--', alpha = 0.5)
ax.set(xlim=(0, max(cells_sample.Ncells)))
plt.savefig(figpath + '/postQC_cells_sample.png', bbox_inches='tight')
plt.clf()

# Extract the samples with < 500 cells, looks at distribution
bad_samps = cells_sample[(cells_sample.Ncells < 500)]
bad_samps =  bad_samps['sample']
bad_samps_proportions = pd.DataFrame(columns=[cats], index=bad_samps)
bad_samps_proportions = bad_samps_proportions.assign(Total=0)

for s in bad_samps:
    use = adata.obs[adata.obs.convoluted_samplename == s]
    bad_samps_proportions.loc[bad_samps_proportions.index == s, 'Total'] = use.shape[0]
    for c in cats:
        prop = use[use.category == c].shape[0]/use.shape[0]
        bad_samps_proportions.loc[bad_samps_proportions.index == s, c] = prop

# Plot the total number of cells in these samples
bad_samps = np.array(bad_samps)
temp = adata.obs[adata.obs.convoluted_samplename.isin(bad_samps)]
temp['convoluted_samplename'] = temp['convoluted_samplename'].astype("string")
temp['convoluted_samplename'] = temp['convoluted_samplename'].astype("category")
plt.figure(figsize=(8, 6))
fig,ax = plt.subplots(figsize=(8,6))
sns.countplot(x=temp["convoluted_samplename"])
#sns.catplot(x="convoluted_samplename", data=adata.obs[adata.obs.convoluted_samplename.isin(bad_samps)])
plt.xlabel('Cells/sample')
ax.set_xticklabels(labels = bad_samps, rotation=45, horizontalalignment='right')
plt.savefig(figpath + '/postQC_total_cells_bad_samples.png', bbox_inches='tight')
plt.clf()

# Plot the depth/cell against sample number per sample (highlighting the high samples)
high_samps = np.array(cells_sample.loc[cells_sample.Ncells > 10000, "sample"])
# Recalculate nGenes/sampls
counts=adata.X
ncounts_gt1=(counts >= 1).sum(axis=1)
adata.obs["ngenes_per_cell"] = ncounts_gt1
# Summarise
depth_count = pd.DataFrame(index = np.unique(adata.obs.convoluted_samplename), columns=["Mean_nCounts", "nCells", "High_cell_sample", "ngenes_per_cell"])
for s in range(0, depth_count.shape[0]):
    samp = depth_count.index[s]
    depth_count.iloc[s,1] = adata.obs[adata.obs.convoluted_samplename == samp].shape[0]
    depth_count.iloc[s,0] = sum(adata.obs[adata.obs.convoluted_samplename == samp].total_counts)/depth_count.iloc[s,1]
    depth_count.iloc[s,3] = sum(adata.obs[adata.obs.convoluted_samplename == samp].ngenes_per_cell)/depth_count.iloc[s,1]
    if samp in high_samps:
        depth_count.iloc[s,2] = "Red"
    else: 
         depth_count.iloc[s,2] = "Navy"
    # Also annotate the samples with low number of cells - Are these sequenced very deeply?
    if samp in bad_samps:
        depth_count.iloc[s,2] = "Green"


depth_count["log10_Mean_Counts"] = np.log10(np.array(depth_count["Mean_nCounts"].values, dtype = "float"))

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(depth_count["Mean_nCounts"], depth_count["ngenes_per_cell"],  c=depth_count["High_cell_sample"], alpha=0.7)
plt.xlabel('Mean counts / cell')
plt.ylabel('Mean genes detected / cell')
plt.savefig(figpath + '/postQC_counts_cells_genes_cells.png', bbox_inches='tight')
plt.clf()

# Have a posthoc check of MT%
depth_count["mean_MT"] = 0
for s in range(0, depth_count.shape[0]):
    samp = depth_count.index[s]
    depth_count.iloc[s,5] = sum(adata.obs[adata.obs.convoluted_samplename == samp].pct_counts_gene_group__mito_transcript)/depth_count.iloc[s,1]

plt.figure(figsize=(8, 6))
plt.scatter(depth_count["Mean_nCounts"], depth_count["mean_MT"],  c=depth_count["High_cell_sample"], alpha=0.7)
plt.xlabel('Mean counts / cell')
plt.ylabel('Mean MT% / cell')
plt.savefig(figpath + '/postQC_counts_cells_mt_perc_cells.png', bbox_inches='tight')
plt.clf()


## Plot distribution of categories per bad sample (Commented out as giving some weird error when submitted and already generated)
#bad_samps_proportions = bad_samps_proportions.drop('Total', axis=1)
#plt.figure(figsize=(16, 12))
#fig,ax = plt.subplots(figsize=(16,12))
#bad_samps_proportions.plot(
#    kind = 'barh',
#    stacked = True,
#    title = 'Stacked Bar Graph',
#    mark_right = True)
#plt.legend(cats, bbox_to_anchor=(1.0, 1.0))
#plt.suptitle('')
#plt.xlabel('Proportion of cell category/sample')
#plt.ylabel('')
#plt.savefig(figpath + '/postQC_prop_cats_bad_samples.png', bbox_inches='tight')
#plt.clf()

# Compare this with a random set of samples
good_samps = cells_sample.sample(n = len(bad_samps))
good_samps =  good_samps['sample']
good_samps_proportions = pd.DataFrame(columns=[cats], index=good_samps)
good_samps_proportions = good_samps_proportions.assign(Total=0)
for s in good_samps:
    use = adata.obs[adata.obs.convoluted_samplename == s]
    good_samps_proportions.loc[good_samps_proportions.index == s, 'Total'] = use.shape[0]
    for c in cats:
        prop = use[use.category == c].shape[0]/use.shape[0]
        good_samps_proportions.loc[good_samps_proportions.index == s, c] = prop

good_samps_proportions = good_samps_proportions.drop('Total', axis=1)
plt.figure(figsize=(16, 12))
fig,ax = plt.subplots(figsize=(16,12))
good_samps_proportions.plot(
    kind = 'barh',
    stacked = True,
    title = 'Stacked Bar Graph',
    mark_right = True)
plt.legend(cats, bbox_to_anchor=(1.0, 1.0))
plt.suptitle('')
plt.xlabel('Proportion of cell category/sample')
plt.ylabel('')
plt.savefig(figpath + '/postQC_prop_cats_random_good_samples.png', bbox_inches='tight')
plt.clf()

# DEBATABLE. DO we remove these samples with < 500 cells/sample and those with > 10k cells/sample?
# Initially had removed these, but the cells from the low number of cell samples look okay - so will keep for atlassing and maybe remove for the eQTL analysis
# For the samples with the very high number of cells / sample: Will keep these initially to see how they integrate, but may repeat and remove these late on

# Following some QC, decided to remove the high cell samples and repeat these
adata = adata[~adata.obs.convoluted_samplename.isin(high_samps)]

# Finally, see considerable intermixing within categories down the line.
# This seems to be occuring between categories, particularly when MT% is high. Interestingly, this is less often the case between epithelial cells with either other 
# Epithelial cells or Immune cells, but instead seems to be occuring between cells labelled as an immune cell category, being placed with epithelial cells. 
# It is possible this is occuring because these are very bad quality immune cells (outlying high MT%), being placed with epithelial cells because these tend to have an 
# inherently high MT%. 
# For this reason, am deciding to define relative cut offs for the percentage of MT gene expression within some resolution of the data
# As we don't want to do this in a way that does not allow admixing within some resolution of the data that may vary between the TI and rectum, will define these 
# relative cut offs within lineages of the data. 
# This will include grouping: 
# 'Stem', 'Enterocyte'and 'Secretory' into 'Epithelial' 
# 'B cell', 'B cell plasma', 'T cell' and 'Myeloid' into 'Immune'
# 'Mesenchymal' into 'Mesenchymal'
# Then defining a relative cut off for these (median + 2.5*MAD), and applying this threshold if it is less than 50% (otherwise will use 50%)
# Add the lineage information to the adata object
lin_df = pd.DataFrame({'category': cats, 'lineage': ""}) 
lin_df['lineage'] = np.where(lin_df['category'].isin(['B Cell', 'B Cell plasma', 'T Cell', 'Myeloid']), 'Immune', lin_df['lineage'])
lin_df['lineage'] = np.where(lin_df['category'].isin(['Stem cells', 'Secretory', 'Enterocyte']), 'Epithelial', lin_df['lineage'])
lin_df['lineage'] = np.where(lin_df['category']== 'Mesenchymal', 'Mesenchymal', lin_df['lineage'])
# Merge onto adata
adata.obs.index = adata.obs.index.astype(str)
cells = adata.obs.index
adata.obs = adata.obs.merge(lin_df, on='category', how="left")
adata.obs.index = cells

#  For each lineage, define the relative cut off and subset a temporary object for these cells
adata_list = []
lins = np.unique(lin_df.lineage)
for index,c in enumerate(lins):
    print(c)
    temp = adata[adata.obs.lineage == c]
    mt_perc=temp.obs["pct_counts_gene_group__mito_transcript"]
    median_plus_2pt5_mad = np.median(mt_perc) + 2.5*(mad(mt_perc))
    print(median_plus_2pt5_mad)
    if median_plus_2pt5_mad < 50:
        temp = temp[temp.obs.pct_counts_gene_group__mito_transcript < median_plus_2pt5_mad]
    adata_list.append(temp)

# Put back together and overwrite the original adata object
adata = ad.concat(adata_list)

# Save this
adata.obs = adata.obs.drop("patient_number", axis=1)
adata.write_h5ad(objpath + "/adata_cell_filt.h5ad")

# 6. Finally, filter for lowly expressed genes
sc.pp.filter_genes(adata, min_cells=6)

print("After removal of the high cell samples, dataset size is")
print(adata.shape)


####################################
##### Expression normalisation #####
####################################

# Before normalising the counts, extract the raw counts and save them
adata.raw = adata
adata.layers['counts'] = adata.X.copy()

# Calulate the CP10K expression
sc.pp.normalize_total(adata, target_sum=1e4)

# Now normalise the data to identify highly variable genes (Same as in Tobi and Monika's paper)
sc.pp.log1p(adata)

# Looks like there are large clumps of cells defined by very high MT%, as opposed to their predicted category from the TI keras annotation (albeit not perfect)
# This seems to dramatically affect the B cell plasma cells
# In addition, IG, RP and MT genes were removed in the TI atlas HVG selection
# RP% doesn't seem to be a problem (plotting pct_counts_gene_group__ribo_protein on umap)
# Initially, 136/228 IG genes are identified as highly variable "IG[HKL]V|IG[KL]J|IG[KL]C|IGH[ADEGM] 
# Doesn't look like IG% or RP% is heavily influencing embedding
# But there are lots of IG genes in the HVG and only one RP
# Will regress out the effect of MT% before scaling, and make sure none of these genes are left in the analysis for batch correction etc

# identify highly variable genes and scale these ahead of PCA
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000, batch_key='convoluted_samplename')

# Check for intersection of IG, MT and RP genes in the HVGs
print("IG")
print(np.unique(adata.var[adata.var.gene_symbols.str.contains("IG[HKL]V|IG[KL]J|IG[KL]C|IGH[ADEGM]")].highly_variable, return_counts=True))
print("MT")
print(np.unique(adata.var[adata.var.gene_symbols.str.contains("^MT-")].highly_variable, return_counts=True))
print("RP")
print(np.unique(adata.var[adata.var.gene_symbols.str.contains("^RP")].highly_variable, return_counts=True))
# MT or RP genes are not present in the HVGs, Lots of IG genes are.
# However on UMAPs in previous attempts, these don't seem to be causing a problem. So will leave them in.

# Scale
sc.pp.scale(adata, max_value=10)

####################################
######### PCA calculation ##########
####################################

# Run PCA
sc.tl.pca(adata, svd_solver='arpack')

# Plot PCA
sc.pl.pca(adata, save="True")

# PLot Elbow plot
sc.pl.pca_variance_ratio(adata, log=True, save=True, n_pcs = 50)

#  Determine the optimimum number of PCs
# Extract PCs
pca_variance=pd.DataFrame({'x':list(range(1, 51, 1)), 'y':adata.uns['pca']['variance']})
# Identify 'knee'
knee=kd.KneeLocator(x=list(range(1, 51, 1)), y=adata.uns['pca']['variance'], curve="convex", direction = "decreasing")
knee_point = knee.knee
elbow_point = knee.elbow
print('Knee: ', knee_point) 
print('Elbow: ', elbow_point)
# Use the 'knee' + 5 additional PCs
nPCs = knee_point + 5
print("The number of PCs used for this analysis is {}".format(nPCs))

# Save the 'knee'
from numpy import asarray
from numpy import savetxt
savetxt(catpath + "/knee.txt", asarray([[knee_point]]), delimiter='\t')

# Save the PCA loadings
loadings = pd.DataFrame(adata.varm['PCs'])
loadings = loadings.iloc[:,0:knee_point]
loadings.index = np.array(adata.var.index)
pcs = list(range(1,knee_point+1))
cols = ["{}{}".format("PC", i) for i in pcs]
loadings.columns = cols
loadings.to_csv(tabdir + "/PCA_loadings.csv")

# Save the PCA matrix 
pca = pd.DataFrame(adata.obsm['X_pca'])
pca = pca.iloc[:,0:knee_point]
pca.index = adata.obs.index
pca.to_csv(tabdir + "/PCA_up_to_knee.csv")

# Save object ahead of batch correction
adata.write(objpath + "/adata_PCAd.h5ad")


###################################
######## Batch correction ##########
####################################

# 1. scVI
# Correcting for MT% (If not done, this seems to be very high in regions of the data where there is overlapping cell categories)
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="experiment_id")
model = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
model.train()
SCVI_LATENT_KEY = "X_scVI"
adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

# Save pre-benchmark
if os.path.exists(objpath) == False:
    os.mkdir(objpath)

adata.write(objpath + "/adata_PCAd_scvid.h5ad")

# 2. scANVI
# Have not performed this on the rectum data as this requires a confident annotation of cells - which is not yet obtained.

# Bench marking of the batch effect correction (using experiment id as I believe convoluted samplename gets re-written at some point)
from scib_metrics.benchmark import Benchmarker
bm = Benchmarker(
    adata,
    batch_key="experiment_id",
    label_key="label",
    embedding_obsm_keys=["X_pca", SCVI_LATENT_KEY],
    n_jobs=1,
    pre_integrated_embedding_obsm_key="X_pca"
)
bm.benchmark()

# Get the results out
df = bm.get_results(min_max_scale=False)
print(df)
# Save 
df.to_csv(tabdir + "/integration_benchmarking.csv")
df1 = df.drop('Metric Type')
top = df1[df1.Total == max(df1.Total.values)].index
print("The method with the greatest overall score is: ")
print(str(top.values))

# Plot the embedding
SCVI_MDE_KEY = "X_scVI_MDE"
adata.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(adata.obsm[SCVI_LATENT_KEY])
sc.pl.embedding(
    adata,
    basis=SCVI_MDE_KEY,
    color=["label", "convoluted_samplename"],
    frameon=False,
    save=True
)

# Compute NN and UMAP using the recommended number of PCs
# Would like to calculate the knee from the latent SCANVI factors, but not sure where this is stored or how to access
# NOTE: Probably want to be doing a sweep of the NN parameter as have done for all of the data together
# Using non-corrected matrix
# Compute UMAP (Will also want to do a sweep of min_dist and spread parameters here)
sc.pp.neighbors(adata, n_neighbors=350, n_pcs=nPCs, use_rep="X_pca", key_added="pca_nn")
sc.tl.umap(adata, neighbors_key="pca_nn", min_dist=0.5, spread=0.5)
sc.pl.umap(adata, color = "label", save="_NN.png")

# Save the ouput adata file
if os.path.exists(objpath) == False:
    os.mkdir(objpath)

adata.write(objpath + "/adata_PCAd_scvid.h5ad")

# Pre-emptively compute the NN embedding using 350 neighbours and custom min_dist and spread parameters
# Used to do this after parameter sweep, but these paramaters usually looked pretty good
knee = pd.read_csv("rectum/" + status + "/" + category + "/knee.txt")
knee = float(knee.columns[0])
knee=int(knee)
nPCs=knee+5
print(nPCs)
# Compute NN
n="350"
n=int(n)
SCVI_LATENT_KEY = "X_scVI"
sc.pp.neighbors(adata, n_neighbors=n, n_pcs=nPCs, use_rep=SCVI_LATENT_KEY, key_added="scVI_nn")
param_sweep_path = "rectum/" + status + "/" + category + "/objects/adata_objs_param_sweep"
nn_file=param_sweep_path + "/NN_{}_scanvi.adata".format(n)
adata.write_h5ad(nn_file)
# Compute UMAP 
sc.tl.umap(adata, min_dist=0.5, spread=0.5, neighbors_key ="scVI_nn")
# Save 
umap_file = re.sub("_scanvi.adata", "_scanvi_umap.adata", nn_file)
adata.write_h5ad(umap_file)


#adata.obs['id_run'] = adata.obs.id_run.astype('category')
adata.obs['convoluted_samplename'] = adata.obs.id_run.astype('category')
#sc.pl.umap(adata, color = "id_run", save="_id_run_NN.png")
sc.pl.umap(adata, color = "convoluted_samplename", save="_sample_NN.png", palette=list(mp.colors.CSS4_COLORS.values()))
# Now using batch effect corrected annotation
sc.pp.neighbors(adata, n_neighbors=350, n_pcs=nPCs, use_rep=SCVI_LATENT_KEY, key_added="scVI_nn")
sc.tl.umap(adata, neighbors_key ="scVI_nn", min_dist=0.5, spread=0.5)
sc.pl.umap(adata, color = "label", save="_" + SCVI_LATENT_KEY + ".png")
sc.pl.umap(adata, color = "convoluted_samplename", save="_sample_" + SCVI_LATENT_KEY + ".png")

# Save the scANVI matrix
scanvi = pd.DataFrame(adata.obsm[SCVI_LATENT_KEY])
scanvi = scanvi.iloc[:,0:knee_point]
scanvi.index = adata.obs.index
scanvi.to_csv(tabdir + "/scVI_up_to_knee.csv")

