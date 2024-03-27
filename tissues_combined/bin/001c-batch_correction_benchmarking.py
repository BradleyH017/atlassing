#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-01-12'
__version__ = '0.0.1'

# Change dir
import os
cwd = os.getcwd()
print(cwd)

# Load packages
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
import scib_metrics 
from scib_metrics.utils import principal_component_regression
print("Loaded libraries")


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
            '-m', '--methods',
            action='store',
            dest='methods',
            required=True,
            help=''
        )
    
    parser.add_argument(
        '-u', '--use_label',
        action='store',
        dest='use_label',
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
    
    parser.add_argument(
        '-vec', '--var_explained_cols',
        action='store',
        dest='var_explained_cols',
        required=True,
        help=''
    )
    
    return parser.parse_args()

def main():
    # Parse options
    inherited_options = parse_options()
    input_file = inherited_options.input_file
    use_label = inherited_options.use_label
    var_explained_cols = inherited_options.var_explained_cols
    var_explained_cols = var_explained_cols.split(",")
    batch_column=inherited_options.batch_column
    
    # Derive and print the tissue arguments
    tissue=inherited_options.tissue
    print(f"~~~~~~~ TISSUE:{tissue}")
    
    # Get the methods
    methods = inherited_options.methods
    methods = methods.split(",")
    print(f"methods: {methods}")
    
    # Do we have a GPU?
    use_gpu = torch.cuda.is_available()
    print(f"Is there a GPU available?: {use_gpu}")
    
    # Load in anndata
    adata = sc.read_h5ad(input_file)
    
    # Load the batch correction matrices and add these to the anndata
    for m in methods:
        print(f"Loading in: {m}")
        sparse = sp.sparse.load_npz(f"results/{tissue}/tables/batch_correction/{m}_matrix.npz")
        dense = sparse.toarray()
        adata.obsm[f"X_{m}"] = dense
    
    obs_keys = ["X_" + method for method in methods]
    obs_keys.insert(0, "X_pca")
    
    
    bm = Benchmarker(
        adata,
        batch_key=batch_column,
        label_key=use_label,
        embedding_obsm_keys=obs_keys,
        n_jobs=2,
        pre_integrated_embedding_obsm_key="X_pca"
    )   
    bm.benchmark()

    # Get the results out
    df = bm.get_results(min_max_scale=False)
    print(df)
    # Save 
    df.to_csv(f"results/{tissue}/tables/batch_correction/benchmark.csv")
    df1 = df.drop('Metric Type')
    top = df1[df1.Total == max(df1.Total.values)].index
    print("The method with the greatest overall score is: ")
    print(str(top.values))
    
    # Save
    with open(f"results/{tissue}/tables/batch_correction/best_batch_method.txt", "w") as file:
        # Write the string to the file
        file.write(str(top.values[0]))    
        
    # Save adata
    adata.write(f"results/{tissue}/objects/adata_PCAd_batched.h5ad")
    
    # Finally, also calculate the variance explained by each numerous variables and how this changes pre/post correction
    var_df = pd.DataFrame(index=var_explained_cols, columns=obs_keys)
    print("Computing variance explained")
    for v in vars_check:
        for m in obs_keys:
            print(f"{v} - {m}")
            var_df.loc[v, m] = scib_metrics.utils.principal_component_regression(adata.obsm[m], adata.obs[v], categorical=True)
            
    var_df.to_csv(f"results/{tissue}/tables/batch_correction/var_explained_pre_post_batch.csv")

    

# Execute
if __name__ == '__main__':
    main()