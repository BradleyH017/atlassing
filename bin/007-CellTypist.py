#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-02-15'
__version__ = '0.0.1'

# Load in packages
import scanpy as sc
import celltypist
from celltypist import models
import anndata as ad
import pandas as pd
import numpy as np
import pandas as pd
import argparse
print("Loaded packages")

# Define the run arguments
def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                QC of tissues together
                """
        )
    
    parser.add_argument(
            '-t', '--tissue',
            action='store',
            dest='tissue',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-h5', '--h5_path',
            action='store',
            dest='h5_path',
            required=True,
            help=''
        )
    
    parser.add_argument(
        '-pm', '--pref_matrix',
        action='store',
        dest='pref_matrix',
        required=True,
        help=''
    )
    
    parser.add_argument(
            '-m', '--model',
            action='store',
            dest='model',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-mn', '--model_name',
            action='store',
            dest='model_name',
            required=True,
            help=''
        )
    
    return parser.parse_args()

# Define main script
def main():
    ########## TESTING #########
    #tissue = "rectum"
    #fpath = f"alternative_results/bi_nmads_results/{tissue}/objects/adata_PCAd_batched_umap_clustered.h5ad" 
    #pref_matrix="X_Harmony"
    #model = "/lustre/scratch127/cellgen/cellgeni/cakirb/celltypist_models/megagut_celltypist_lowerGI+lym_adult.pkl"
    #model_name = "Megagut"
    ###########################
    inherited_options = parse_options()
    tissue = inherited_options.tissue
    fpath = inherited_options.h5_path
    use_matrix_f = inherited_options.pref_matrix
    with open(use_matrix_f, 'r') as file:
        pref_matrix = file.read().strip()
    
    model = inherited_options.model
    #model = model.split(",")
    print(f"model: {model}")
    model_name = inherited_options.model_name
    #model_name = model_name.split(",")
    print(f"model_name: {model_name}")

    # Load in the data to be annotated (cell filtered, batches
    adata = ad.read_h5ad(fpath)

    # Re-log1p CP10K the data
    adata.X = adata.layers['log1p_cp10k'].copy()

    # Use gene_symbols
    adata.X.index = adata.var.gene_symbols
    adata.var.index = adata.var.gene_symbols

    # Copy the desired nearest neighbor matrix NN so that it is used in the model
    adata.uns['neighbors'] = adata.uns[f"{pref_matrix}_nn"]
    adata.obsp['connectivities'] = adata.obsp[f"{pref_matrix}_nn_connectivities"]
    adata.obsp['distances'] = adata.obsp[f"{pref_matrix}_nn_distances"]
    if "leiden" in adata.obs.columns:
        adata.obs['cluster'] = adata.obs['leiden']
    
    # NOTE: TO DO: Also adjust X_scVI_nn_connectivities

    # Run the model (majority voting)
    predictions = celltypist.annotate(adata, model = model, majority_voting = True)

    # Transform the anndata object to include these annotations
    ct_adata = predictions.to_adata()

    # Perform dotplot
    sc.settings.figdir=f"results/{tissue}/figures/annotation"
    celltypist.dotplot(predictions, use_as_reference = 'cluster', use_as_prediction = 'majority_voting', filter_prediction=0.01, save=f"dotplot_majority_voting_{model_name}_label__machine.png")
    celltypist.dotplot(predictions, use_as_reference = 'cluster', use_as_prediction = 'predicted_labels', filter_prediction=0.01, save=f"dotplot_predicted_labels_{model_name}_label__machine.png")

    # Save the predictions, decision matrix
    probmat = predictions.probability_matrix
    probmat["CellTypist_model"] = model_name
    probmat.to_csv(f"results/{tissue}/tables/annotation/CellTypist/CellTypist_prob_matrix.csv")
    decmat = predictions.decision_matrix
    decmat["CellTypist_model"] = model_name
    decmat.to_csv(f"results/{tissue}/tables/annotation/CellTypist/CellTypist_decision_matrix.csv")

    # Plot on UMAP
    sc.settings.figdir=f"results/{tissue}/figures/UMAP/annotation"
    adata.obsm["X_umap"] = adata.obsm["UMAP_" + pref_matrix]
    sc.pl.umap(ct_adata, color="majority_voting", frameon=True, save=f"_{model_name}_majority_voting_{pref_matrix}.png")

    # Save prediction
    to_save = adata.obs[["predicted_labels", "over_clustering", "majority_voting", "conf_score"]]
    to_save['CellTypist_model'] = model_name
    to_save.to_csv(f"results/{tissue}/tables/annotation/CellTypist/CellTypist_anno_conf.csv")
    
if __name__ == '__main__':
    main()