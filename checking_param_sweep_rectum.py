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
sys.path.append('/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/conda_envs/scRNAseq/CellRegMap')
from cellregmap import run_association, run_interaction, estimate_betas
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
sc.pl.umap(adata, color="id_run",frameon=True, save="_post_batch_post_sweep_id_run.png")

# Rename convoluted sample name (this seems to have been lost after the batch correction). 
# Have checked the adata used for batch correction and this is correct though. So not to worry.
adata.obs["convoluted_samplename"] = adata.obs['convoluted_samplename'].str.replace("__donor", '')

# Check where the high samples look here
high_cell_samps = ["OTARscRNA13669852", "OTARscRNA13430444"]
def check_intersection(value):
    if value in high_cell_samps:
        return 'yes'
    else:
        return 'no'

# Create a new column 'intersects' using the apply method
adata.obs['high_cell_samp'] = adata.obs['convoluted_samplename'].apply(lambda x: check_intersection(x))

# Plot the high cell samples on UMAP
sc.pl.umap(adata, color="high_cell_samp",frameon=True, save="_post_batch_post_sweep_high_n_cells.png")

# Which id_run are these samples from? 46765 and 47258
test = adata.obs[["convoluted_samplename", "id_run"]]
test = test.reset_index()
test = test[["convoluted_samplename", "id_run"]]
test = test.drop_duplicates()
test[test.convoluted_samplename.isin(high_cell_samps)]

bad_int = test[test.convoluted_samplename.isin(high_cell_samps)].id_run.values
test[test.id_run.isin(bad_int)]

# Assessment of batch effect integration
df = pd.read_csv(data_name + "/" + status + "/" + category + "/tables/integration_benchmarking.csv")



