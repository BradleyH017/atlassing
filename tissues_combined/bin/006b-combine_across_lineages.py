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
import gc
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
            '-bo', '--baseout',
            action='store',
            dest='baseout',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-bto5', '--base_tissue_original_h5',
            action='store',
            dest='base_tissue_original_h5',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-t', '--tissues',
            action='store',
            dest='tissues',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-rbl', '--remove_blacklist',
            action='store',
            dest='remove_blacklist',
            required=True,
            help=''
        )
    
    return parser.parse_args()
    

def main():
    inherited_options = parse_options()
    baseout = inherited_options.baseout
    base_tissue_original_h5 = inherited_options.base_tissue_original_h5
    tissues = inherited_options.tissues.split(",")
    remove_blacklist = inherited_options.remove_blacklist
    print(f"baseout: {baseout}")
    print(f"base_tissue_original_h5: {base_tissue_original_h5}")
    print(f"tissues: {tissues}")
    print(f"remove_blacklist: {remove_blacklist}")

    # For each lineage, read in the h5ad, extract the cells and clusters, add to a single dataframe
    all_clusters = []
    blacklist_samps = []
    for t in tissues:
        print(f"Loading QC optimal clusters for {t}")
        temp = sc.read_h5ad(f"results/{t}/objects/adata_clusters_post_clusterQC.h5ad")
        print(temp.shape)
        if "cell" not in temp.obs.columns:
            temp.obs.reset_index(inplace=True)
    
        clusters = temp.obs[["cell", "leiden"]]
        if len(all_clusters) == 0:
            all_clusters = clusters
        else:
            all_clusters = pd.concat([all_clusters, clusters])
        
        del temp
        gc.collect()
        # If they exist, also read in the blacklist samples
        bfile = f"results/{t}/figures/clustering_array_summary/black_list_samples_postQC.csv"
        if os.path.exists(bfile):
            print(f"Loading blacklist samples for {t}")
            blacklist = pd.read_csv(bfile, index_col=0)
            blacklist['tissue'] = t
            if len(blacklist_samps) == 0:
                blacklist_samps = blacklist
            else:
                blacklist_samps = pd.concat([blacklist_samps, blacklist])
        
    # Print the blacklist samples
    print("blacklist_samps:")
    print(blacklist_samps)
    
    # Read in the original adata
    adata = sc.read_h5ad(base_tissue_original_h5)
    print(f"Shape of original adata: {adata.shape}")
    
    # Remove the blacklist samples if desired. NOTE: This will remove blacklist samples derived from each lineage, from ALL lineages. So if one sample has strange epithelial cells, it's mesenchymal and immune cells will also be removed
    orig = adata.shape[0]
    if remove_blacklist == "yes":
        adata = adata[~adata.obs['samp_tissue'].isin(blacklist_samps['samp_tissue'])]   
        
    # Subset for the cells in the final clusters, then merge
    adata.obs.reset_index(inplace=True)
    adata.obs = adata.obs.merge(all_clusters, on="cell", how="left")
    adata.obs.set_index("cell", inplace=True)

    # Subset to only inlcude the cells in clusters we kept after QC
    adata = adata[adata.obs['leiden'].notna()]

    # Print the loss of cells
    new = adata.shape[0]
    diff = orig-new
    perc = 100*diff/orig
    print(f"After within lineage QC and cluster QC, lost {orig-new} ({perc:.2g}%) cells")
        
    # Add lineage if not present
    if "manual_lineage" not in adata.obs.columns:
        adata.obs['manual_lineage'] = adata.obs['leiden'].str.split('_').str[0]    
    
    # Save
    adata.write_h5ad(f"results/{baseout}/objects/adata_grouped_post_cluster_QC.h5ad")

    # Plot
    #sc.settings.figdir=f"results/{baseout}/figures/UMAP/annotation"
    #sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
    #sc.logging.print_header()
    #sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")
    #sc.pl.umap(adata, color="leiden", save="_filtered_reproducible_clusters_all.png")
    

# execute
if __name__ == '__main__':
    main()
    
    
    