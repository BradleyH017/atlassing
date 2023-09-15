##### Bradley July 2023
##### Sweep of kNN, min_dist and spread paramateres for clustering and embedding of rectum scRNAseq data
##### conda activate scvi-env

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

# Define data_name
data_name="rectum"
status="healthy"
category="All_not_gt_subset"
param_sweep_path = data_name + "/" + status + "/" + category + "/objects/adata_objs_param_sweep"

# Retreive options
n = sys.argv[1] #kNN
print("The number of NN for this analysis is {}".format(n))

# Check if this NN file has already been created on this data
nn_file=param_sweep_path + "/NN_{}_scanvi.adata".format(n)
if os.path.isfile(nn_file):
    print("NN file exists, so loading this in")
    adata = ad.read_h5ad(nn_file)
    print("Loaded NN file")
else:
    print("NN file does not exist so loading the PCAd file")
    # Load the PCAd/scVId data
    adata=ad.read_h5ad(data_name + "/" + status + "/" + category + "/objects/adata_PCAd_scvid.h5ad")
    print("Loaded")
    # Load the knee
    knee = pd.read_csv(data_name + "/" + status + "/" + category + "/knee.txt")
    knee = float(knee.columns[0])
    knee=int(knee)
    nPCs=knee+5
    print(nPCs)
    # Compute NN
    n=int(n)
    SCVI_LATENT_KEY = "X_scVI"
    sc.pp.neighbors(adata, n_neighbors=n, n_pcs=nPCs, use_rep=SCVI_LATENT_KEY, key_added="scVI_nn")
    print("Computed NN")
    # Save
    adata.write(nn_file)
    print("Saved NN file")

# Compute the UMAP in a for loop over paramaters
for m in (0.1, 0.3, 0.5, 1, 2):
    for s in (0.1, 0.5, 1, 2, 3):
        # Re-compute UMAP using these parameters
        print("Computing UMAP for Min_dist={}, spread={}".format(m,s))
        sc.tl.umap(adata, min_dist=m, spread=s, neighbors_key ="scVI_nn")
        print("Computed UMAP")
        # Plot UMAP
        save_str = "{}nn_{}_min_dist_{}_spread.pdf".format(n,m,s)
        sc.settings.figdir=data_name + "/" + status + "/" + category + "/figures/embedding_param_sweep"
        sc.pl.umap(adata, color="category",frameon=False, title="NN={}, min_dist={}, spread={}".format(n, m, s), save=save_str)
        print("Plotted UMAP")