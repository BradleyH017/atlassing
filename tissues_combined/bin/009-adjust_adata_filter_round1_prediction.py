#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-02-15'
__version__ = '0.0.1'

# Load in packages
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import pandas as pd
import argparse
import os
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
            '-h5', '--h5ad',
            action='store',
            dest='h5ad',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-cts', '--col_to_sort',
            action='store',
            dest='col_to_sort',
            required=False,
            help=''
        )
    
    parser.add_argument(
        '-of', '--output_file',
        action='store',
        dest='output_file',
        help='Basename of output files, assuming output in current working directory.'
    )
    
    return parser.parse_args()


def main():
    inherited_options = parse_options()
    h5ad = inherited_options.h5ad
    col_to_sort = inherited_options.col_to_sort
    output_file = inherited_options.output_file
    # TESTING
    # h5ad="temp/base-.h5ad"
    # col_to_sort="predicted_celltype"
    # output_file="temp/round1-"
    
    # Load data
    adata = sc.read_h5ad(h5ad)

    # Divide by the value of col_to_sort
    groups = np.unique(adata.obs[col_to_sort])
    for g in groups:
        print(g)
        # Filter
        print("Filtering")
        temp = adata[adata.obs[col_to_sort] == g]    
        print(f"{temp.shape[0]} cells")
        # Adjust column names to reflect round (if not already)
        is_round_present = any('round' in col_name for col_name in adata.obs.columns)
        if not is_round_present:
            print("Renaming cols")
            for col_name in temp.obs.columns:
                if 'predicted' in col_name:
                    temp.obs.rename(columns={col_name: 'round1__' + col_name}, inplace=True)
        
        # Remove "celltype__"
        g = g.replace("celltype__", "")
        
        # Save
        print("saving")
        temp.write_h5ad(f"{output_file}{g}.h5ad")
        

# Execute            
if __name__ == '__main__':
    main()
