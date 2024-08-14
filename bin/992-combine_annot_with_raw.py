#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-05-03'
__version__ = '0.0.1' 

# Import libraries
import scanpy as sc
import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
from sklearn.decomposition import PCA
import plotnine as plt9
import argparse
from numpy import asarray
print("Loaded packages")


# Define the run arguments
def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                collection of multiple adatas
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
            '-orig', '--orig',
            action='store',
            dest='orig',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-fc', '--filtered_clustered',
            action='store',
            dest='filtered_clustered',
            required=True,
            help=''
        )
    
    return parser.parse_args()
    

def main():
    inherited_options = parse_options()
    tissue = inherited_options.tissue
    original_h5ad = inherited_options.orig
    filtered_clustered = inherited_options.filtered_clustered
    
    # Load in the filtered_clustered file
    fc = sc.read_h5ad(filtered_clustered)
    fc.obs.reset_index(inplace=True)
    to_add = fc.obs[["cell", "leiden"]]
    print(f"Ncells of the clustered data: {to_add.shape[0]}")
    del fc
    
    # Now load the original
    adata = sc.read_h5ad(original_h5ad)
    adata.obs.reset_index(inplace=True)
    adata.obs = adata.obs.merge(to_add, on="cell", how="left")
    
    # Remove cells without confident annotation
    adata = adata[adata.obs['leiden'].notna()]
    adata.obs.set_index("cell", inplace = True)
    print(f"Ncells of the appended original adata: {adata.shape[0]}")
    
    # Save
    adata.write_h5ad(f"results/{tissue}/objects/adata_raw_confident_cluster_cells_annotated.h5ad")
    

# execute
if __name__ == '__main__':
    main()