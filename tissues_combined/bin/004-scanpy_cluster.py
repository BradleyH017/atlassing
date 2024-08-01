#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-02-13'
__version__ = '0.0.1'

# Load packages
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
import matplotlib as mp
import argparse

# Def options parsing 
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
        '-f', '--fpath',
        action='store',
        dest='fpath',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-cr', '--clustering_resolution',
        action='store',
        dest='clustering_resolution',
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
    
    return parser.parse_args()


def main():
    inherited_options = parse_options()
    tissue = inherited_options.tissue
    print(f"tissue: {tissue}")
    # Note: This requires the bknn conda env
    fpath = inherited_options.fpath
    print(f"fpath: {fpath}")
    clustering_resolution = float(inherited_options.clustering_resolution)
    print(f"clustering_resolution: {clustering_resolution}")
    use_matrix_f = inherited_options.pref_matrix
    with open(use_matrix_f, 'r') as file:
        pref_matrix = file.read().strip() # Should be a single value
    print(f"pref_matrix: {pref_matrix}")

    # Load the data
    adata = sc.read_h5ad(fpath)    
    
    # Redefine the NN matrix (from the one derived from the desired latent variables)
    pref_matrix_nn = pref_matrix + "_nn"
    
    # Cluster using optimum NN and the given resolution
    sc.tl.leiden(adata, resolution=clustering_resolution, neighbors_key=pref_matrix_nn)
    
    # Save adata
    adata.write(f"results/{tissue}/tables/clustering_array/leiden_{clustering_resolution}/adata_PCAd_batched_umap_{clustering_resolution}.h5ad")
    
    # Extract the cell x cluster matrix
    annot = adata.obs[['leiden']]
    annot.columns = annot.columns + "_" + str(clustering_resolution)
    annot.reset_index(inplace=True)
    
    # write to file
    annot.to_csv(f"results/{tissue}/tables/clustering_array/leiden_{str(clustering_resolution)}/clusters.csv", index=False)
    
    # Plot on umap
    sc.settings.figdir=f"results/{tissue}/tables/clustering_array/leiden_{str(clustering_resolution)}"
    sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")
    sc.pl.umap(adata, color = "leiden", save=f"_clusters_res{clustering_resolution}.png")


# Execute
if __name__ == '__main__':
    main()  