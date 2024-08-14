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
#from scib_metrics.benchmark import Benchmarker
from pytorch_lightning import Trainer
import argparse
import scipy as sp
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
            '-pm', '--pref_matrix',
            action='store',
            dest='pref_matrix',
            required=True,
            help=''
        )
    
    return parser.parse_args()

def main():
    # Parse options
    inherited_options = parse_options()
    input_file = inherited_options.input_file
    tissue = inherited_options.tissue
    pref_matrix = inherited_options.pref_matrix
    
    # Load anndata
    adata = sc.read_h5ad(input_file)
    
    # Load matrix
    print(f"Loading in: {pref_matrix}")
    sparse = sp.sparse.load_npz(f"results/{tissue}/tables/batch_correction/{pref_matrix}_matrix.npz")
    dense = sparse.toarray()
    adata.obsm[f"X_{pref_matrix}"] = dense
    
    # Save 
    adata.write(f"results/{tissue}/objects/adata_PCAd_batched.h5ad")
    

# Execute
if __name__ == '__main__':
    main()