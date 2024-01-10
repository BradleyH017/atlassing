#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-01-010'
__version__ = '0.0.1'

# This is the step performs the reference mapping of the query dataset and label transfer from reference to query
# Following example using the HLCA - https://docs.scarches.org/en/latest/hlca_map_classify.html
# Needs to be in scarches environment

# Load libraries
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import scanpy as sc
import torch
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt
import numpy as np
import gdown
import scvi
from scarches.models.scpoli import scPoli
# Forked from https://github.com/MarioniLab/oor_benchmark.git
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import argparse
from scipy import sparse
from sklearn.metrics import classification_report
import anndata as ad
absolute_path = str(Path("repos/oor_benchmark/src/oor_benchmark/methods").resolve())
sys.path.append(absolute_path)
print("Loaded packages")

# Parse script options

# Parse script options (ref, query, outdir)
def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                Transfer learning of reference annotations to the query
                """
        )
    
    parser.add_argument(
            '-r', '--scANVI_ref__file',
            action='store',
            dest='scANVI_ref__file',
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
            '-m', '--ref_model_dir',
            action='store',
            dest='ref_model_dir',
            required=True,
            help=''
        )

    parser.add_argument(
            '-q', '--query_data',
            action='store',
            dest='query_data',
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

# Define the main script
def main():
    inherited_options = parse_options()
    scANVI_ref__file = inherited_options.scANVI_ref__file
    # scANVI_ref__file="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/rectum_to_TI/objects/adata_ref_scANVI.h5ad"
    ref_model_dir = inherited_options.ref_model_dir
    # ref_model_dir = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/rectum_to_TI/objects/TI_scanvi_model"
    query_data = inherited_options.query_data
    # query_data = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/rectum/healthy/All_not_gt_subset/objects/adata_PCAd_batched.h5ad"
    label_col = inherited_options.label_col
    # label_col="label__machine"
    outdir = inherited_options.outdir
    # outdir = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/rectum_to_TI/objects"
    sample_identifier = inherited_options.sample_identifier
    # sample_identifier = "sanger_sample_id"

    # Load the reference scANVI object
    ref_adata = sc.read_h5ad(scANVI_ref__file)

    # Load the query 
    adata_query_unprep = sc.read_h5ad(query_data)

    # Prep data for scArches
    adata_query = sca.models.SCANVI.prepare_query_anndata(
        adata=adata_query_unprep, reference_model=ref_model_dir, inplace=False
    )

    # Load the reference model to perform surgery on
    surgery_model = sca.models.SCANVI.load_query_data(
        adata_query,
        ref_model_dir,
        freeze_dropout=True,
    )

