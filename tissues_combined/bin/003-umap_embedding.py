#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-01-12'
__version__ = '0.0.1'

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
import matplotlib as mp
import argparse

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
        '-u', '--use_matrix',
        action='store',
        dest='use_matrix',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-k', '--knee_file',
        action='store',
        dest='knee_file',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-nn', '--optimum_nn_file',
        action='store',
        dest='optimum_nn_file',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-cb', '--col_by',
        action='store',
        dest='col_by',
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
    knee_file = inherited_options.knee_file
    print(f"knee_file: {knee_file}")
    use_matrix = inherited_options.use_matrix
    use_matrix = use_matrix.split(",")
    use_matrix = ["X_" + sub for sub in use_matrix] 
    print(f"use_matrix: {use_matrix}")
    optimum_nn_file = inherited_options.optimum_nn_file
    print(f"optimum_nn_file: {optimum_nn_file}")
    col_by = inherited_options.col_by
    print(f"col_by: {col_by}")
    
    # Load in the anndata (batch corrected)
    adata = sc.read_h5ad(fpath)
    
    # Load the knee and optimum NN params
    with open(knee_file, 'r') as file:
        content = file.read().strip()
        n_pcs = int(float(content)) + 5
        print(f"n_pcs: {n_pcs}")
        
    with open(optimum_nn_file, 'r') as file:
        content = file.read().strip()
        optimum_nn = int(float(content))
        print(f"optimum_nn: {optimum_nn}")
        
    # Define plot outdir
    figpath = f"results/{tissue}/figures/UMAP/"
    if os.path.exists(figpath) == False:
        os.mkdir(figpath)

    # Define plot options
    sc.settings.figdir=figpath
    sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")
    
    # Define outdir for gene expression plots
    exprfigpath = f"results/{tissue}/figures/UMAP/expr"
    if os.path.exists(exprfigpath) == False:
        os.mkdir(exprfigpath)
    
    # Now compute UMAP for each embedding using these parameters
    use_matrix=np.intersect1d(list(adata.obsm.keys()), use_matrix)
    col_by = col_by.split(",")
    print(f"col_by: {col_by}")
    print(f"figpath: {figpath}")
    # Calculate NN
    for m in use_matrix:
        print("Calculating neighbours")
        sc.pp.neighbors(adata, n_neighbors=optimum_nn, n_pcs=n_pcs, use_rep=m, key_added= m + "_nn")
        # Perform UMAP embedding
        sc.tl.umap(adata, neighbors_key=m + "_nn", min_dist=0.5, spread=0.5)
        adata.obsm["UMAP_" + m] = adata.obsm["X_umap"]
        
        # Save a plot
        for c in col_by:
            print(c)
            if c in adata.var['gene_symbols'].values:
                    ens=adata.var[adata.var['gene_symbols'] == c].index[0]
                    sc.settings.figdir=exprfigpath
                    sc.pl.umap(adata, color = ens, save="_" + m + "_" + c + ".png")
            elif c == "experiment_id":
                    sc.settings.figdir=figpath
                    sc.pl.umap(adata, color = c, save="_" + m + "_" + c + ".png", palette=list(mp.colors.CSS4_COLORS.values()))
                
            elif c in adata.obs.columns:
                    sc.settings.figdir=figpath
                    sc.pl.umap(adata, color = c, save="_" + m + "_" + c + ".png")

    # Overwite file after each 
    adata.write(f"results/{tissue}/objects/adata_PCAd_batched_umap.h5ad")
    
    
# Execute
if __name__ == '__main__':
    main()  