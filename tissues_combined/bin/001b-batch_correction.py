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
        '-bc', '--batch_correction',
        action='store',
        dest='batch_correction',
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
        '-sacol', '--scANVI_col',
        action='store',
        dest='scANVI_col',
        required=False,
        help=''
    )
    
    parser.add_argument(
        '-cvo', '--correct_variable_only',
        action='store',
        dest='correct_variable_only',
        required=False,
        help=''
    )

    return parser.parse_args()

def main():
    # Parse options
    inherited_options = parse_options()
    input_file = inherited_options.input_file
    batch_correction=inherited_options.batch_correction
    batch_correction=batch_correction.split("|")
    batch_column=inherited_options.batch_column
    correct_variable_only=inherited_options.correct_variable_only
    
    # Derive and print the tissue arguments
    tissue=inherited_options.tissue
    print(f"~~~~~~~ TISSUE:{tissue}")
    
    # Do we have a GPU?
    use_gpu = torch.cuda.is_available()
    print(f"Is there a GPU available?: {use_gpu}")
    
    # Load in anndata (backed)
    adata = sc.read_h5ad(input_file)
    
    # Get basedir - modified to run in current dir
    # outdir = os.path.dirname(os.path.commonprefix([input_file]))
    # outdir = os.path.dirname(os.path.commonprefix([outdir]))

    print("Set up outdirs")
    
    if correct_variable_only:
            adata = adata[:,adata.var['highly_variable'] == True].copy()
    
    if "scVI" in batch_correction:
        # 1. scVI
        print("~~~~~~~~~~~~~~~~~~~ Batch correcting with scVI - optimum  ~~~~~~~~~~~~~~~~~~~")
        #Trainer(accelerator="cuda")
        # See is a GPU is available - if so, use. If not, then adjust
        scvi.settings.dl_pin_memory_gpu_training =  use_gpu
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_column)
        model = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
        model.train(use_gpu=use_gpu)
        SCVI_LATENT_KEY = "X_scVI"
        adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()
        sparse_matrix = sp.sparse.csc_matrix(adata.obsm[SCVI_LATENT_KEY])
        sp.sparse.save_npz(f"results/{tissue}/tables/batch_correction/scVI_matrix.npz", sparse_matrix)
    
    
    if "scVI_default" in batch_correction:
        # 2. scVI - default_metrics
        print("~~~~~~~~~~~~~~~~~~~ Batch correcting with scVI - Default  ~~~~~~~~~~~~~~~~~~~")
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_column)
        model_default = scvi.model.SCVI(adata,  n_latent=30)
        model_default.train(use_gpu=use_gpu)
        SCVI_LATENT_KEY_DEFAULT = "X_scVI_default"
        adata.obsm[SCVI_LATENT_KEY_DEFAULT] = model_default.get_latent_representation()
        sparse_matrix = sp.sparse.csc_matrix(adata.obsm[SCVI_LATENT_KEY_DEFAULT])
        sp.sparse.save_npz(f"results/{tissue}/tables/batch_correction/scVI_default_matrix.npz", sparse_matrix)

    if "Harmony" in batch_correction:
        print("~~~~~~~~~~~~~~~~~~~ Batch correcting with Harmony ~~~~~~~~~~~~~~~~~~~")
        sc.external.pp.harmony_integrate(adata, batch_column, basis='X_pca', adjusted_basis='X_Harmony')
        sparse_matrix = sp.sparse.csc_matrix(adata.obsm['X_Harmony'])
        sp.sparse.save_npz(f"results/{tissue}/tables/batch_correction/Harmony_matrix.npz", sparse_matrix)
    
    if "scANVI" in batch_correction:
        # 3. scANVI
        print("~~~~~~~~~~~~~~~~~~~ Batch correcting with scANVI ~~~~~~~~~~~~~~~~~~~")
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_column)
        model = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
        scanvi_model = scvi.model.SCANVI.from_scvi_model(
            model,
            adata=adata,
            labels_key=inherited_options.scANVI_col,
            unlabeled_category="Unknown",
        )
        scanvi_model.train(max_epochs=20, n_samples_per_label=100, use_gpu=use_gpu)
        SCANVI_LATENT_KEY = "X_scANVI"
        adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)
        sparse_matrix = sp.sparse.csc_matrix(adata.obsm[SCANVI_LATENT_KEY])
        sp.sparse.save_npz(f"results/{tissue}/tables/batch_correction/scANVI_matrix.npz", sparse_matrix)
    

# Execute
if __name__ == '__main__':
    main()