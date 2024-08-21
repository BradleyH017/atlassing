#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-08-21'
__version__ = '0.0.1' 

import pandas as pd
import numpy as np
import scanpy as sc
import celltypist
import argparse
import os


def main():
    """Run CLI."""
    parser = argparse.ArgumentParser(
        description="""
            Assesses celltypist autoannotation on the original input data
            """
    )
    parser.add_argument(
        '-h5', '--h5_anndata',
        action='store',
        dest='h5',
        required=True,
        help='Path to h5 AnnData file.'
    )
    
    parser.add_argument(
        '-ctm', '--celltypist_model',
        action='store',
        dest='celltypist_model',
        required=True,
        help='celltypist model'
    )
    
    parser.add_argument(
        '-o', '--outdir',
        action='store',
        dest='outdir',
        default='.',
        required=False,
        help='Output directory'
    )  
    
    options = parser.parse_args()
    h5 = options.h5
    celltypist_model = options.celltypist_model
    outdir = options.outdir

    # Test
    # h5 = "results_round2/all_Mesenchymal/objects/adata_clusters_post_clusterQC.h5ad"
    # celltypist_model = "results/combined/objects/leiden-adata_grouped_post_cluster_QC.pkl"
    # outdir = "temp"

    # Load the AnnData file.
    print('Loading AnnData')
    adata = sc.read_h5ad(filename=h5)
    
    # use log1p_cp10k
    adata.X = adata.layers['log1p_cp10k']

    # Use gene_symbols
    adata.var['ENS'] = adata.var_names
    adata.var_names = list(adata.var['gene_symbols'])
    adata.var_names_make_unique()
    
    # Run the model (majority voting)
    predictions = celltypist.annotate(adata, model = celltypist_model, majority_voting = False)
    ct_adata = predictions.to_adata()

    # Perform dotplot
    sc.settings.figdir=outdir
    celltypist.dotplot(predictions, use_as_reference = 'leiden', use_as_prediction = 'predicted_labels', filter_prediction=0.01, save=f"_predicted_labels_vs_leiden_autoannot.png")

    # Save the celltypist output
    ct_adata.write_h5ad(f"{outdir}/celltypist_prediction.h5ad")
    

# Execute 
if __name__ == '__main__':
    main()
    