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
from anndata.experimental import read_elem
from h5py import File
print("Loaded libraries")

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
        '-sc', '--samp_col',
        action='store',
        dest='samp_col',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-of', '--output_file',
        action='store',
        dest='output_file',
        required=True,
        help=''
    )
    
    return parser.parse_args()

def main():    
    inherited_options = parse_options()
    h5ad = inherited_options.h5ad
    samp_col = inherited_options.samp_col
    output_file = inherited_options.output_file
    print("parsed args")
    
    # Load in the data (obs only)
    f2 = File(h5ad, 'r')
    obs = read_elem(f2['obs'])
    
    # In a for loop, divide by sample and save to the output file
    groups = np.unique(obs[samp_col])
    
    # Save a list of samples
    with open(f"{output_file}sample_list.txt", 'w') as file:
        for g in groups:
            print(g)
            file.write(str(g) + '\n')


# Execute
if __name__ == '__main__':
    main()