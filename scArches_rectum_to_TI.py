#!/usr/bin/env python

# Script to compute an scvi model of one tissue (TI) in order to map a second tissue (rectum) onto this data
# Following https://github.com/theislab/scarches/blob/master/notebooks/scvi_surgery_pipeline.ipynb and
# https://github.com/theislab/scarches/blob/master/notebooks/hlca_map_classify.ipynb and 
# https://github.com/MarioniLab/oor_benchmark/tree/master/src/oor_benchmark/methods
__author__ = 'Bradley Harris'
__date__ = '2024-01-02'
__version__ = '0.0.1'

## Load packagesimport scanpy as sc
# git clone https://github.com/theislab/scarches
# cd scarches
# mamba env create -f envs/scarches_linux.yaml
# mamba activate scarches
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
                Prepping files for the SAIGEQTL analysis
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
    query__file = inherited_options.query__file
    outdir = inherited_options.output_directory
    
    # Load in the reference data (TI)
    adata = sc.read(ref__file)
    
    # define the plot outdir
    plotdir=f"{outdir}/plots"
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
        
    
    sc.settings.figdir=plotdir
    
    # determine the batch and label keys
    batch_key="sanger_sample_id"
    label_key="label"
    
    # Create scvi model using the query data labels
    sca.models.SCVI.setup_anndata(adata, batch_key=batch_key, labels_key=label_key)

    # Train this on the reference data - NOTE: This is using the optimum parameters suggested by the authors: https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/harmonization.html
    vae = sca.models.SCVI(
        adata,
        n_layers=2,
        n_latent=30,
        gene_likelihood="nb",
        encode_covariates=True,
        deeply_inject_covariates=False,
        use_layer_norm="both",
        use_batch_norm="none",
    )

    print("Training the reference scVI model")
    vae.train()

    # Get latent representation of this data (NOTE: not really neccessary, but will compute anyway)
    print("Getting latent representation of the reference model")
    reference_latent = sc.AnnData(vae.get_latent_representation())
    reference_latent.obs["cell_type"] = adata.obs[label_key].tolist()
    reference_latent.obs["batch"] = adata.obs[batch_key].tolist()
    sc.pp.neighbors(reference_latent, n_neighbors=200)
    sc.tl.leiden(reference_latent)
    sc.tl.umap(reference_latent)
    sc.pl.umap(reference_latent,
            color=['batch', 'cell_type'],
            frameon=False,
            wspace=0.6,
            save="ref_latent.png")
    
    # Save the reference model to output
    ref_path = f"{outdir}/ref_model"
    vae.save(ref_path, overwrite=True)
    print("Saved the reference model")
    
    # Now load in the query data
    target_adata = sc.read(query__file)
    
    # Perform surgery on the reference and training
    model = sca.models.SCVI.load_query_data(
        target_adata,
        ref_path,
        freeze_dropout = True,
    )
    
    # Train the model
    print("Training the model on the query dataset")
    model.train(max_epochs=200, plan_kwargs=dict(weight_decay=0.0))

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

    
    
    
    
    
    
    
    
    
    
    
    
    
    


