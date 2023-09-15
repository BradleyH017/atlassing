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

# Check if this NN file has already been created on this data
nn_file=param_sweep_path + "/NN_{}_scanvi.adata".format(n)
adata = ad.read_h5ad(nn_file)

# Recompute UMAP with percieved optimum conditions
sc.tl.umap(adata, min_dist=min_dist, spread=spread, neighbors_key ="scVI_nn")

# Define fig out dir
sc.settings.figdir=data_name + "/" + status + "/" + category + "/figures"

# Plot categories, labels, samples
sc.pl.umap(adata, color="category",frameon=True, save="_post_batch_post_sweep_category.png")
sc.pl.umap(adata, color="label",frameon=True, save="_post_batch_post_sweep_keras.png")
sc.pl.umap(adata, color="bead_lot",frameon=True, save="_post_batch_post_sweep_bead_lot.png")
sc.pl.umap(adata, color="gem_lot",frameon=True, save="_post_batch_post_sweep_gem_lot.png")

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



