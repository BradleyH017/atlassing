#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-01-12'
__version__ = '0.0.1'

# Load packages
import scanpy as sc
import scipy as sp
import pandas as pd
import numpy as np
import scib_metrics
from scib_metrics import kbet
import seaborn as sns
import matplotlib.pyplot as plt
import scib as scib
import argparse
from numpy import asarray
from numpy import savetxt

# Parse script options (ref, query, outdir)
def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                Batch correction of tissues together
                """
        )
    
    parser.add_argument(
            '-i', '--input_file',
            action='store',
            dest='input_file',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-tissue', '--tissue',
            action='store',
            dest='tissue',
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
        '-k', '--knee_file',
        action='store',
        dest='knee_file',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-nn', '--nn_val',
        action='store',
        dest='nn_val',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-col', '--batch_column',
        action='store',
        dest='batch_column',
        required=True,
        help=''
    )
    
    return parser.parse_args()


# Define main script
def main():
    # Parse options
    inherited_options = parse_options()
    input_file = inherited_options.input_file
    tissue = inherited_options.tissue
    pref_matrix = inherited_options.pref_matrix
    nn = int(inherited_options.nn)
    knee_file = inherited_options.knee_file
    batch_column = inherited_options.batch_column
    
    # Load adata
    adata = sc.read_h5ad(input_file)
    
    # Load knee
    with open(knee_file, 'r') as file:
        content = file.read().strip()
        n_pcs = int(float(content)) + 5
        print(f"n_pcs: {n_pcs}")
    
    # Compute NN on the desired matrix
    print("Calculating neighbours")
    sc.pp.neighbors(adata, n_neighbors=nn, n_pcs=n_pcs, use_rep="X_" + pref_matrix, key_added= "X_" + pref_matrix + "_nn")
    
    # Extract this matrix
    neighbors_matrix = adata.obsp["X_" + pref_matrix + "_nn"]
    
    # Extract sample identifier
    sample_ids = adata.obs[batch_column]
    
    # Initialize a dictionary to hold the count of cells from each sample per neighbor group
    sample_mixing = {}
    print("Calculating sample mixing")
    for i in range(neighbors_matrix.shape[0]):
        # Get indices of neighbors for the i-th cell
        neighbor_indices = neighbors_matrix[i].nonzero()[1]
        # Get the sample ids of these neighbors
        neighbor_samples = sample_ids.iloc[neighbor_indices]
        # Get the ref sample
        ref_sample = sample_ids[i]
        # Count the number of times cells from the same sample are observed within the nearest neighbours
        same_sample_count = sum(neighbor_samples == ref_sample)
        # Store the result
        sample_mixing[i] = same_sample_count
    
    # Convert to DataFrame for easier manipulation and visualization
    sample_mixing_df = pd.DataFrame(list(sample_mixing.items()), columns=['cell', 'Nsame_samp'])
    sample_mixing_df['cell'] = adata.obs.index
    sample_mixing_df['prop_same_samp'] = sample_mixing_df['Nsame_samp'] / nn
    
    mean_samp_exclusivity = sum(sample_mixing_df['prop_same_samp']) / sample_mixing_df.shape[0]
    max_samp_exclusivity = max(sample_mixing_df['prop_same_samp'])
    
    # Now compute the iLISI
    adata.uns['neighbors'] = adata.uns["X_" + pref_matrix + "_nn"]
    adata.obsp['connectivities'] = adata.obsp["X_" + pref_matrix + "_nn" + "_connectivities"] 
    adata.obsp['distances'] = adata.obsp["X_" + pref_matrix + "_nn" + "_distances"] 
    ilisi = scib.metrics.ilisi_graph(adata, batch_key=batch_column, type_="knn", k0=nn, scale=True)
    
    # Rather than saving these seperately, save as one file
    data = {"parameter": ["mean_exclusivity", "max_exclusivity", "median_ilisi"],
            "value": [mean_samp_exclusivity, max_samp_exclusivity, ilisi]}
    data_df = pd.DataFrame(data)
    data_df['nn'] = nn
    data_df.to_csv(f"results/tissue/tables/nn_array/{nn}_summary.csv", header=False)
    
    
if __name__ == '__main__':
    main()