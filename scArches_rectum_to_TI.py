#!/usr/bin/env python

# Script to compute an scvi model of one tissue (TI) in order to map a second tissue (rectum) onto this data
# Following https://github.com/theislab/scarches/blob/master/notebooks/scvi_surgery_pipeline.ipynb and
# https://github.com/theislab/scarches/blob/master/notebooks/hlca_map_classify.ipynb and 
# https://github.com/MarioniLab/oor_benchmark/tree/master/src/oor_benchmark/methods
# Forked from https://github.com/MarioniLab/oor_benchmark.git
__author__ = 'Bradley Harris'
__date__ = '2024-01-02'
__version__ = '0.0.1'

## Load packagesimport scanpy as sc
# Using the atlassing conda environment
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import scanpy as sc
import torch
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt
import numpy as np
import gdown
import scvi
from scarches.models.scpoli import scPoli
# Forked from https://github.com/MarioniLab/oor_benchmark.git
from pathlib import Path
import sys
absolute_path = str(Path("repos/oor_benchmark/src/oor_benchmark/methods").resolve())
sys.path.append(absolute_path)
from _latent_embedding import embedding_scArches, _filter_genes_scvi, _fit_scVI, _train_scVI

print("Loaded libraries")

# Set script and plotting params
sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

# Parse arguements
def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                Transfer learning of TI annotations into the rectum
                """
        )
    
    parser.add_argument(
            '-r', '--ref__file',
            action='store',
            dest='query__file',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-q', '--query__file',
            action='store',
            dest='query__file',
            required=True,
            help=''
        )

    parser.add_argument(
            '-o', '--output_directory',
            action='store',
            dest='output_directory',
            required=True,
            help=''
        )



# Define the main script
def main():
    inherited_options = parse_options()
    ref__file = inherited_options.ref__file
    # ref__file = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/freeze_003/ti-cd_healthy-fr003_004/anderson_ti_freeze003_004-eqtl_processed.h5ad"
    # Think it is worth making sure this is the VERY GOOD QUALITY CELLS FROM DISCOVERY AND REPLICATION
    query__file = inherited_options.query__file
    # query__file="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/2023_09_rectum/adata.h5ad"
    # All cells
    outdir = inherited_options.output_directory
    # outdir = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/rectum_to_TI"
    
    # Load in the reference data (TI)
ref = sc.read(ref__file)
ref.layers["counts"] = ref.X.copy()

# Load in the query
query = sc.read(query__file)
query.layers["counts"] = query.X.copy()

# define the plot outdir
plotdir=f"{outdir}/plots"
if not os.path.exists(plotdir):
    os.makedirs(plotdir)
    

sc.settings.figdir=plotdir

# determine the batch and label keys
batch_key="sanger_sample_id"
label_key="label"

# Generate an scVI model on the reference data
sca.models.SCVI.setup_anndata(ref, batch_key=batch_key)

# Add params (NOTE: These are not default, but recommended by scVI)
vae = sca.models.SCVI(
    ref,
    n_layers=2,
    n_latent=30,
    gene_likelihood="nb",
    encode_covariates=True,
    deeply_inject_covariates=False,
    use_layer_norm="both",
    use_batch_norm="none",
)

vae.train()

# Plot before saving
latent_key='X_scVI'
ref[latent_key] = sc.AnnData(vae.get_latent_representation())
ref.obs["cell_type"] = ref.obs[label_key].tolist()
ref.obs["batch"] = ref.obs[batch_key].tolist()
sc.pp.neighbors(ref, n_neighbors=300, use_rep=latent_key)
sc.tl.umap(ref)
sc.pl.umap(ref,
           color=['batch', 'cell_type', 'category__machine'],
           frameon=False,
           wspace=0.6,
           )

# Save this
ref_path = f"{outdir}/ref_model"
vae.save(ref_path, overwrite=True)

# Now transfer the query dataset into this space using scArches
model = sca.models.SCVI.load_query_data(
    query,
    ref_path,
    freeze_dropout = True,
)

model.train(max_epochs=200, plan_kwargs=dict(weight_decay=0.0))

query[latent_key] = sc.AnnData(model.get_latent_representation())
query.obs['cell_type'] = query.obs[label_key].tolist()
query.obs['batch'] = query.obs[batch_key].tolist()





    # Get the latent representation of the query dataset (Also probably not all that interesting, but will generate anyway)
    query_latent = sc.AnnData(model.get_latent_representation())
    query_latent.obs['cell_type'] = target_adata.obs[cell_type_key].tolist()
    query_latent.obs['batch'] = target_adata.obs[condition_key].tolist()
    sc.pp.neighbors(query_latent)
    sc.tl.leiden(query_latent)
    sc.tl.umap(query_latent)
    plt.figure()
    sc.pl.umap(
        query_latent,
        color=["batch", "cell_type"],
        frameon=False,
        wspace=0.6,
    )

    
    
    
    
    
    
    
    
    
    
    
    
    
    


