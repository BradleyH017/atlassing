#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-01-12'
__version__ = '0.0.1'

# Import libraries
import sys
print("System path")
print(f"sys.path: {sys.path}")
import os
conda_prefix = os.environ.get('CONDA_PREFIX')
print(f"conda prefix = {conda_prefix}")
# sys.path = f"{conda_prefix}/lib/python3.10/site-packages/"
print(f"New sys.path: {sys.path}")
import scanpy as sc
import bbknn as bbknn
from numpy import asarray
from numpy import savetxt
import argparse

def parse_options():    
    parser = argparse.ArgumentParser(
        description="""
            Compute optimum NN from the desired matrix with bbknn
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
        '-k', '--knee_file',
        action='store',
        dest='knee_file',
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
    print(f"use_matrix: {use_matrix}")

    # Load in the anndata (post QC)
    adata = sc.read_h5ad(fpath)

    # Load the knee (and add 5)
    with open(knee_file, 'r') as file:
        content = file.read().strip()
        n_pcs = int(float(content)) + 5
        print(n_pcs)

    # Batch correct 
    # NOTE: ************** (params neighbors_within_batch is for testing only) ***********
    bbknn.bbknn(adata, batch_key='experiment_id', n_pcs=n_pcs, use_rep=use_matrix)

    # Extract the optimum NN defined by this and save
    optimum_nn = adata.uns['neighbors']['params']['n_neighbors']

    # Save the optimum NN
    savetxt(f"results/{tissue}/tables/optimum_nn.txt", asarray([[optimum_nn]]), delimiter='\t')


# Execute
if __name__ == '__main__':
    main()
    






