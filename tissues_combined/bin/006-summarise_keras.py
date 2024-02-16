#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-02-15'
__version__ = '0.0.1'
# Script has been taken from yascp (https://github.com/wtsi-hgi/yascp/blob/1d465f848a700b9d39c4c9a63356bd7b985f5a1c/bin/0057-scanpy_cluster_validate_resolution-keras.py#L4)

# Load in packages
import sys
sys.path.append('/software/team152/bh18/pip')
sys.path.append('/usr/local/')
print("System path")
print(sys.path)
import numpy as np
print("Numpy file")
print(np.__file__)
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib as mp
from matplotlib import pyplot as plt
from matplotlib.pyplot import rc_context
import kneed as kd
import scvi
import csv
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import math
import scipy.stats as st
import re
from scipy import stats
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from sympy import symbols, Eq, solve
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fsolve
from scipy.optimize import brentq
import torch
from scib_metrics.benchmark import Benchmarker
from pytorch_lightning import Trainer
import argparse
import scipy as sp
import plotnine as plt9
import glob
from numpy import asarray
from numpy import savetxt
print("Loaded libraries")

# Define custom plotting function (from https://github.com/wtsi-hgi/yascp/blob/1d465f848a700b9d39c4c9a63356bd7b985f5a1c/bin/0058-plot_resolution_boxplot.py):
def _make_plots(
    df_plt,
    out_file_base,
    y='AUC',
    facet_grid='',
    h_line=''
):
    len_x = len(np.unique(df_plt['resolution']))
    if 'sparsity_l1' in df_plt.columns:
        df_plt['Sparsity'] = df_plt['sparsity_l1']
        len_x2 = len(np.unique(df_plt['Sparsity']))
    else:
        len_x2 = 0
    if len_x2 > 1:
        gplt = plt9.ggplot(df_plt, plt9.aes(
            fill='Sparsity',
            x='resolution',
            y=y,
        ))
        gplt = gplt + plt9.geom_boxplot(
            alpha=0.8,
            outlier_alpha=0
        )
        gplt = gplt + plt9.geom_jitter(
            plt9.aes(color='Sparsity'),
            alpha=0.25,
            width=0.2
        )
    else:
        gplt = plt9.ggplot(df_plt, plt9.aes(
            x='resolution',
            y=y
        ))
        gplt = gplt + plt9.geom_boxplot(
            alpha=0.8,
            outlier_alpha=0
        )
        gplt = gplt + plt9.geom_jitter(
            alpha=0.25,
            width=0.2
        )
    gplt = gplt + plt9.theme_bw(base_size=12)
    if facet_grid != '':
        gplt = gplt + plt9.facet_grid('{} ~ .'.format(facet_grid))
    if y == 'f1-score':
        gplt = gplt + plt9.labs(
            x='Resolution',
            y='F1 score',
            title=''
        )
    elif y in ['AUC', 'MCC']:
        gplt = gplt + plt9.labs(
            x='Resolution',
            y=y,
            title=''
        )
    else:
        gplt = gplt + plt9.labs(
            x='Resolution',
            y=y.capitalize().replace('_', ' '),
            title=''
        )
    gplt = gplt + plt9.theme(
        # legend_position='none',
        axis_text_x=plt9.element_text(angle=-45, hjust=0)
    )
    if len_x2 != 0 and len_x2 < 9:
        gplt = gplt + plt9.scale_fill_brewer(
            palette='Dark2',
            type='qual'
        )
    if h_line != '':
        gplt = gplt + plt9.geom_hline(
            plt9.aes(yintercept=h_line),
            linetype='dashdot'
        )
    gplt.save(
        '{}-resolution__{}.png'.format(out_file_base, y.replace('-', '_')),
        #dpi=300,
        width=4*((len_x+len_x2)/4),
        height=5,
        limitsize=False
    )

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
            '-o', '--outpath',
            action='store',
            dest='outpath',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-m', '--MCC_thresh',
            action='store',
            dest='MCC_thresh',
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
    fpath = inherited_options.h5_path
    outpath = inherited_options.outpath
    MCC_thresh = float(inherited_options.MCC_thresh)
    pref_matrix = inherited_options.pref_matrix

    ########## TESTING #########
    #tissue = "blood"
    #fpath = f"results/{tissue}/objects/adata_PCAd_batched_umap.h5ad" 
    #outpath = f"results/{tissue}/figures/clustering_array_summary/keras_accuracy"
    #MCC_thresh = 0.5
    #pref_matrix = "X_pca_harmony"
    ###########################

    # Read in all the clustering array results
    dfs = []

    # Define a function to extract resolution from a file path
    def extract_resolution(file_path):
        return file_path.split('/')[-2].split('_')[-1]

    # Use glob to find all matching file paths
    file_paths = glob.glob(f"results/{tissue}/tables/clustering_array/leiden_*/base-model_report.tsv.gz")

    # Loop through each file path
    for file_path in file_paths:
        # Read the file into a DataFrame
        df = pd.read_csv(file_path, sep='\t', compression='gzip')
        # Extract resolution from the file path
        resolution = extract_resolution(file_path)
        # Append the resolution as a new column
        df['resolution'] = resolution
        # Append the DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    # Remove accuracy, macro_avg, weighted_avg
    combined_df = combined_df[~combined_df['cell_label'].isin(['accuracy', 'macro avg', 'weighted avg'])]


    # Columns to plot
    cols_plot = [
            'precision', 'recall', 'f1-score', 'support', 'AUC',
            'average_precision_score', 'MCC'
        ]

    # Make the plots
    for i in cols_plot:
        _make_plots(
                combined_df,
                outpath,
                y=i,
                facet_grid='',
                h_line=MCC_thresh
                )

    # Define what the optimum resolution is. Save this to a file
    combined_df['pass_mcc'] = combined_df['MCC'] > MCC_thresh
    all_mcc_pass = combined_df.groupby('resolution')['pass_mcc'].all()
    optim_resolution = max(all_mcc_pass[all_mcc_pass == True].index)
    savetxt(f"results/{tissue}/tables/optim_resolution.txt", asarray([[float(optim_resolution)]]), delimiter='\t')

    # Load the cluster annotations: 
    clusters = pd.read_csv(f"results/{tissue}/tables/clustering_array/leiden_{optim_resolution}/clusters.csv")

    # Load the anndata object and append
    adata = sc.read_h5ad(fpath)
    adata.obs = adata.obs.reset_index()
    adata.obs = adata.obs.merge(clusters, on = "cell", how="left")
    adata.obs.set_index("cell", inplace=True)
    adata.obs.rename(columns = {f"leiden_{optim_resolution}": "cluster"}, inplace=True)
    adata.obs['cluster'] = adata.obs['cluster'].astype('category')

    # Plot the clusters using the preferred matrix
    sc.settings.figdir=f"results/{tissue}/figures/UMAP/annotation/"
    sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")
    adata.obsm['X_umap'] = adata.obsm["UMAP_" + pref_matrix].copy()
    sc.pl.umap(adata, color = "cluster", save="_" + pref_matrix + "NN_optim_clusters.png")

    # Find the markers of these clusters and save
    sc.tl.rank_genes_groups(adata, 'cluster', method='wilcoxon', key_added="cluster_markers", layer="log1p_cp10k")
    markers = []
    for c in np.unique(adata.obs['cluster'].astype("category")):
        df = sc.get.rank_genes_groups_df(adata, group=str(c), key='cluster_markers')
        df['cluster'] = str(c)
        markers.append(df)

    markers_all = pd.concat(markers)

    # Append gene symbols before saving
    conv = adata.var['gene_symbols']
    conv = conv.reset_index()
    conv.rename(columns={"index": "names"}, inplace=True)
    markers_all = markers_all.merge(conv, how="left", on="names")
    markers_all.to_csv(f"results/{tissue}/tables/markers/markers_all_optim_clusters.txt.gz", sep = "\t", compression='gzip')

    # Make a plot of these
    sc.settings.figdir=f"results/{tissue}/figures/markers/"
    sc.pl.rank_genes_groups(adata, n_genes=25, gene_symbols="gene_symbols", key="cluster_markers", sharey=False, save="_markers_all.png")

    ## TO DO: Get the probabilities of all cells
    #test_results = pd.read_csv(f"results/{tissue}/tables/clustering_array/leiden_{optim_resolution}/base-test_result.tsv.gz", sep='\t', compression='gzip')
    #test_results.rename(columns={col: col.replace('class', 'probability') for col in test_results.columns}, inplace=True)
    #test_results.rename(columns={"cell_label_predicted": "Keras"}, inplace=True)

    # Save the clustered object
    adata.write(f"results/{tissue}/objects/adata_PCAd_batched_umap_clustered.h5ad")

    
if __name__ == '__main__':
    main()