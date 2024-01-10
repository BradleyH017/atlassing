#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-01-05'
__version__ = '0.0.1'

# Perform scANVI batch integration of the TI data (all)
# This is the first step to perform reference mapping and label transfer
# Following example using the HLCA - https://docs.scarches.org/en/latest/hlca_map_classify.html

# Load libraries
import sys
import pandas as pd
#sys.path.append('/software/team152/bh18/pip')
#sys.path.append('/usr/local/')
print("System path")
print(sys.path)
import numpy as np
print("Numpy file")
print(np.__file__)
import anndata as ad
import matplotlib as mp
from matplotlib import pyplot as plt
from matplotlib.pyplot import rc_context
#import kneed as kd
import scvi
import csv
import datetime
#import seaborn as sns
import matplotlib.pyplot as plt
import math
import scipy.stats as st
import re
from scipy import stats
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
# from sympy import symbols, Eq, solve
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fsolve
from scipy.optimize import brentq
import torch
from pytorch_lightning import Trainer
scvi.settings.dl_pin_memory_gpu_training =  True
from scipy import sparse
import argparse
print("Loaded libraries")

# Plotting options
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

# Parse script options (ref, query, outdir)
def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                scANVI model computation of the the reference dataset
                """
        )
    
    parser.add_argument(
            '-r', '--ref__file',
            action='store',
            dest='ref__file',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-l', '--label_col',
            action='store',
            dest='label_col',
            required=True,
            help=''
        )

    parser.add_argument(
            '-o', '--outdir',
            action='store',
            dest='outdir',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-s', '--sample_identifier',
            action='store',
            dest='sample_identifier',
            required=True,
            help=''
        )

    return parser.parse_args()


def main():
    inherited_options = parse_options()
    ref__file = inherited_options.scANVI_ref__file
    # ref__file="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/freeze_003/ti-cd_healthy-fr003_004/anderson_ti_freeze003_004-eqtl_processed.h5ad"
    label_col = inherited_options.label_col
    # label_col="label__machine"
    outdir = inherited_options.outdir
    # outdir = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/rectum_to_TI/objects"
    sample_identifier = inherited_options.sample_identifier
    # sample_identifier = "sanger_sample_id"
    
    # Load in the anndata
    adata = ad.read_h5ad(ref__file)
    print("Loaded the data")
    print(f"Dataset shape: {adata.shape}")

    # Check if GPU is available:
    gpu_use = torch.cuda.is_available()
    print(f"Is a GPU avalable?: {gpu_use}")

    # Make sure there is a 'counts' layer for 'X', and this is raw (printing)
    adata.layers['counts'] = adata.X.copy()
    print(sparse.csr_matrix(adata.X)[:10, :30].toarray())

    # Make sure label_col is in the adata.obs
    if not label_col in adata.obs.columns:
        annot = pd.read_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/highQC_TI_discovery/data-clean_annotation_full.csv")
        annot['cluster'] = annot['cluster'].astype('category')
        adata.obs['cluster'] = adata.obs['cluster'].astype(int)
        annot['label__machine'] = annot['label__machine_retired'] # Use correct version
        cells = adata.obs.index
        adata.obs = adata.obs.merge(annot, on="cluster")
        adata.obs.index = cells
        
    # Build scVI model using GPU
    print("~~~~~~~~~~~~~~~~~~~ Batch correcting with scVI - optimum params ~~~~~~~~~~~~~~~~~~~")
    #Trainer(accelerator="cuda")
    scvi.settings.dl_pin_memory_gpu_training =  True
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=sample_identifier)
    model = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
    model.train(use_gpu=True)
    SCVI_LATENT_KEY = "X_scVI"
    adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

    # Then run scANVI on this model
    # Performing this across lineage
    print("~~~~~~~~~~~~~~~~~~~ Batch correcting with scANVI ~~~~~~~~~~~~~~~~~~~")
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        model,
        adata=adata,
        labels_key=label_col,
        unlabeled_category="Unknown",
    )
        
    #scanvi_model.train(max_epochs=20, n_samples_per_label=100, use_gpu=True)
    scanvi_model.train(max_epochs=20, n_samples_per_label=100)
    SCANVI_LATENT_KEY = "X_scANVI"
    adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)
        
    # Save the scanvi model and the adata object with these reductions
    adata.obs.drop('cluster', axis=1, inplace=True)
    adata.write(outdir + "/adata_ref_scANVI.h5ad")
    scanvi_model.save(f"{outdir}/TI_scanvi_model", overwrite=True)
    
# Execute
if __name__ == '__main__':
    main()
    
