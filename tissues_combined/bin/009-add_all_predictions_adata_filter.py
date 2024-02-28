#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-02-28'
__version__ = '0.0.1'

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
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
        '-pf', '--prediction_file',
        action='store',
        dest='prediction_file',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-pt', '--probability_threshold',
        action='store',
        dest='probability_threshold',
        required=True,
        help=''
    )
    
    return parser.parse_args()


def main():
    inherited_options = parse_options()
    tissue = inherited_options.tissue
    fpath = inherited_options.fpath
    prediction_file = inherited_options.prediction_file
    probability_threshold = inherited_options.probability_threshold
    
    # Load in the raw anndata object
    adata = ad.read_h5ad(fpath)
    print(f"Initial adata shape: {adata.shape}")
    
    # Load the predictions
    predictions = pd.read_csv(prediction_file, sep = "\t", index_col=0)
    
    # Combine 
    adata.obs = adata.obs.merge(predictions, left_index=True, right_index=True)
    
    # Filter if desired
    adata = adata[adata.obs["predicted_celltype_probability"] > float(probability_threshold)]
    print(f"Filtered adata shape: {adata.shape}")
    
    # Save
    adata.write(f"results/{tissue}/objects/adata_raw_predicted_celltypes_filtered.h5ad")
    
    
# Execute
if __name__ == '__main__':
    main()  
    
    