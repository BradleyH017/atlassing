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
        '-bbc', '--benchmark_batch_correction',
        action='store',
        dest='benchmark_batch_correction',
        required=True,
        help=''
    )

    return parser.parse_args()

def main():
    # Parse options
    inherited_options = parse_options()
    input_file = inherited_options.input_file
    batch_correction=inherited_options.batch_correction
    batch_correction=batch_correction.split("|")
    benchmark_batch_correction=inherited_options.benchmark_batch_correction
    
    # Derive and print the tissue arguments
    tissue=inherited_options.tissue
    print(f"~~~~~~~ TISSUE:{tissue}")
    
    # Do we have a GPU?
    use_gpu = torch.cuda.is_available()
    print(f"Is there a GPU available?: {use_gpu}")
    
    # Load in anndata (backed)
    adata = sc.read_h5ad(input_file, backed="r")
    
    # Get basedir - modified to run in current dir
    # outdir = os.path.dirname(os.path.commonprefix([input_file]))
    # outdir = os.path.dirname(os.path.commonprefix([outdir]))
    outdir = "results"
    if os.path.exists(outdir) == False:
        os.mkdir(outdir)
    
    # Update the outdir
    outdir = f"{outdir}/{tissue}"
    objpath = f"{outdir}/objects"
    tabpath = f"{outdir}/tables"

    print("Set up outdirs")
    
    if "scVI" in batch_correction:
        # 1. scVI
        print("~~~~~~~~~~~~~~~~~~~ Batch correcting with scVI - optimum  ~~~~~~~~~~~~~~~~~~~")
        #Trainer(accelerator="cuda")
        # See is a GPU is available - if so, use. If not, then adjust
        scvi.settings.dl_pin_memory_gpu_training =  use_gpu
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="samp_tissue")
        model = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
        model.train(use_gpu=use_gpu)
        SCVI_LATENT_KEY = "X_scVI"
        adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()
    
    
    if "scVI_default" in batch_correction:
        # 2. scVI - default_metrics
        print("~~~~~~~~~~~~~~~~~~~ Batch correcting with scVI - Default  ~~~~~~~~~~~~~~~~~~~")
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="samp_tissue")
        model_default = scvi.model.SCVI(adata,  n_latent=30)
        model_default.train(use_gpu=use_gpu)
        SCVI_LATENT_KEY_DEFAULT = "X_scVI_default"
        adata.obsm[SCVI_LATENT_KEY_DEFAULT] = model_default.get_latent_representation()
    
    if "scANVI" in batch_correction:
        # 3. scANVI
        # Performing this across lineage
        print("~~~~~~~~~~~~~~~~~~~ Batch correcting with scANVI ~~~~~~~~~~~~~~~~~~~")
        scanvi_model = scvi.model.SCANVI.from_scvi_model(
            model,
            adata=adata,
            labels_key=lineage_column,
            unlabeled_category="Unknown",
        )
        scanvi_model.train(max_epochs=20, n_samples_per_label=100, use_gpu=use_gpu)
        SCANVI_LATENT_KEY = "X_scANVI"
        adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)

    if "Harmony" in batch_correction:
        print("~~~~~~~~~~~~~~~~~~~ Batch correcting with Harmony ~~~~~~~~~~~~~~~~~~~")
        sc.external.pp.harmony_integrate(adata, 'samp_tissue', basis='X_pca', adjusted_basis='X_Harmony')

    # Save
    if os.path.exists(objpath) == False:
        os.mkdir(objpath)

    adata.write(objpath + "/adata_PCAd_batched.h5ad")
    
    if benchmark_batch_correction == "yes":
        bm = Benchmarker(
            adata,
            batch_key="samp_tissue",
            label_key="label__machine",
            embedding_obsm_keys=["X_pca", SCVI_LATENT_KEY, SCVI_LATENT_KEY_DEFAULT, SCANVI_LATENT_KEY, 'X_pca_harmony'],
            n_jobs=4,
            pre_integrated_embedding_obsm_key="X_pca"
        )   
        bm.benchmark()

        # Get the results out
        df = bm.get_results(min_max_scale=False)
        print(df)
        # Save 
        df.to_csv(tabpath + "/integration_benchmarking.csv")
        df1 = df.drop('Metric Type')
        top = df1[df1.Total == max(df1.Total.values)].index
        print("The method with the greatest overall score is: ")
        print(str(top.values))
        
    

# Execute
if __name__ == '__main__':
    main()