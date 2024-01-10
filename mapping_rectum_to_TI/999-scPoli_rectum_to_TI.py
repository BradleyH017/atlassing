#!/usr/bin/env python

# Script to compute an scvi model of one tissue (TI) in order to map a second tissue (rectum) onto this data
# Following https://github.com/theislab/scarches/blob/master/notebooks/scvi_surgery_pipeline.ipynb and
# https://github.com/theislab/scarches/blob/master/notebooks/hlca_map_classify.ipynb and 
# https://github.com/MarioniLab/oor_benchmark/tree/master/src/oor_benchmark/methods
# Forked from https://github.com/MarioniLab/oor_benchmark.git
__author__ = 'Bradley Harris'
__date__ = '2024-01-02'
__version__ = '0.0.1'

## Load packagesimport scanpy as sc
# Using the scarches (!!!!) conda environment
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
#from _latent_embedding import embedding_scArches, _filter_genes_scvi, _fit_scVI, _train_scVI

print("Loaded libraries")

# Set script and plotting params
sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

# Parse arguements
def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                Transfer learning of TI annotations into the rectum
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
            '-q', '--query__file',
            action='store',
            dest='query__file',
            required=True,
            help=''
        )

    parser.add_argument(
            '-o', '--output_directory',
            action='store',
            dest='output_directory',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-m', '--method_hvgs',
            action='store',
            dest='method_hvgs',
            required=True,
            help=''
        )

    return parser.parse_args()

# Define the main script
def main():
    inherited_options = parse_options()
    ref__file = inherited_options.ref__file
    # ref__file = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/freeze_003/ti-cd_healthy-fr003_004/anderson_ti_freeze003_004-eqtl_processed.h5ad"
    query__file = inherited_options.query__file
    # query__file="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/2023_09_rectum/adata.h5ad"
    # All cells
    outdir = inherited_options.output_directory
    # outdir = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/rectum_to_TI"
    method_hvgs = inherited_options.method_hvgs
    print("~~~~~~~~ Printing options ~~~~~~~~~~")
    print(f"ref__file: {ref__file}")
    print(f"query__file: {query__file}")
    print(f"output_directory: {outdir}")
    print(f"method_hvgs: {method_hvgs}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Make outdir if not already
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Load in the reference data (TI)
    source_adata = sc.read(ref__file)
    source_adata.layers["counts"] = source_adata.X.copy()
    print("Loaded source data")
    print(f"source shape = {source_adata.shape}")

    # Annotate these
    annot = pd.read_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/highQC_TI_discovery/data-clean_annotation_full.csv")
    annot['cluster'] = annot['cluster'].astype('category')
    source_adata.obs['cluster'] = source_adata.obs['cluster'].astype(int)
    cells = source_adata.obs.index
    source_adata.obs = source_adata.obs.merge(annot, on="cluster")
    source_adata.obs.index=cells
    source_adata.obs['label__machine'] = source_adata.obs['label__machine_retired']

    # Load in the query
    target_adata = sc.read(query__file)
    # Replace 'X' with raw data
    target_adata.X = target_adata.layers['counts'].copy()
    # Make sure column names match
    target_adata.obs = target_adata.obs.rename(columns = {"convoluted_samplename": "sanger_sample_id"})
    target_adata.obs = target_adata.obs.rename(columns = {"Keras:predicted_celltype": "label__machine"})
    print("Loaded target data")
    print(f"target shape = {target_adata.shape}")

    # Calculate the highly variable of the source
    sc.pp.normalize_total(source_adata, target_sum=1e4)
    sc.pp.log1p(source_adata)
    sc.pp.highly_variable_genes(source_adata, flavor="seurat", n_top_genes=4000, batch_key='sanger_sample_id', subset=True)

    # Prep the targetdata
    sc.pp.normalize_total(target_adata, target_sum=1e4)
    sc.pp.log1p(target_adata)

    # get into the same HVGs, depending on method
    if method_hvgs == "reference":
        # Subset the target for these genes
        target_adata = target_adata[:,target_adata.var.index.isin(source_adata.var.index)]
        # Incase some are missed, remove these from the reference
        source_adata = source_adata[:,source_adata.var.index.isin(target_adata.var.index)]

    # Make sure we are using the raw counts for the analysis
    source_adata.X = source_adata.layers['counts'].copy()
    target_adata.X = target_adata.layers['counts'].copy()

    # define the plot outdir
    plotdir=f"{outdir}/plots"
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
        

    sc.settings.figdir=plotdir

    # determine the batch and label keys
    condition_key="sanger_sample_id"
    cell_type_key="label__machine"

    # Use scPoli to integrate query into the reference and perform transfer learning
    # From https://github.com/theislab/scarches/blob/master/notebooks/scpoli_surgery_pipeline.ipynb
    # Better: https://docs.scarches.org/en/latest/scpoli_surgery_pipeline.html
    # NOTE: These are using the paramaters of this github and have not been optimised
    scpoli_model = scPoli(
        adata=source_adata,
        condition_keys=condition_key,
        cell_type_keys=cell_type_key,
        embedding_dims=5,
        recon_loss='nb',
    )

    early_stopping_kwargs = {
        "early_stopping_metric": "val_prototype_loss",
        "mode": "min",
        "threshold": 0,
        "patience": 20,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }

    print("~~~~~~~~~ training the scPoli model ~~~~~~~~~~~~")
    scpoli_model.train(
        n_epochs=50,
        pretraining_epochs=40,
        early_stopping_kwargs=early_stopping_kwargs,
        eta=5,
    )

    # Now map the query into this space
    scpoli_query = scPoli.load_query_data(
        adata=target_adata,
        reference_model=scpoli_model,
        labeled_indices=[],
    )

    print("~~~~~~~~~ Mapping query into reference ~~~~~~~~~~~~")
    # Train the model
    scpoli_query.train(
        n_epochs=50,
        pretraining_epochs=40,
        eta=10
    )
        
    # Transfer the labels from reference to query
    results_dict = scpoli_query.classify(target_adata, scale_uncertainties=True)
    for i in range(len(cell_type_key)):
        preds = results_dict[cell_type_key]["preds"]
        results_dict[cell_type_key]["uncert"]
        classification_df = pd.DataFrame(
            classification_report(
                y_true=target_adata.obs[cell_type_key],
                y_pred=preds,
                output_dict=True,
            )
        ).transpose()

    print(classification_df)
    tabdir = f"{outdir}/tables"
    if not os.path.exists(tabdir):
        os.makedirs(tabdir)
        
    classification_df.to_csv(f"{tabdir}/scpoli_classification_report.csv")
        
    # get latent representation of reference data
    print("~~~~~~~~~~~~ Calculating latent representations ~~~~~~~~~~~~")
    scpoli_query.model.eval()
    data_latent_source = scpoli_query.get_latent(
        source_adata, 
        mean=True
    )

    adata_latent_source = sc.AnnData(data_latent_source)
    adata_latent_source.obs = source_adata.obs.copy()

    #get latent representation of query data
    data_latent= scpoli_query.get_latent(
        target_adata, 
        mean=True
    )

    adata_latent = sc.AnnData(data_latent)
    adata_latent.obs = target_adata.obs.copy()

    #get label annotations
    adata_latent.obs['cell_type_pred'] = results_dict['label__machine']['preds'].tolist()
    adata_latent.obs['cell_type_uncert'] = results_dict['label__machine']['uncert'].tolist()
    adata_latent.obs['classifier_outcome'] = (
        adata_latent.obs['cell_type_pred'] == adata_latent.obs['label__machine']
    )

    #get prototypes
    labeled_prototypes = scpoli_query.get_prototypes_info()
    labeled_prototypes.obs['study'] = 'labeled prototype'
    unlabeled_prototypes = scpoli_query.get_prototypes_info(prototype_set='unlabeled')
    unlabeled_prototypes.obs['study'] = 'unlabeled prototype'

    #join adatas
    adata_latent_full = adata_latent_source.concatenate(
        [adata_latent, labeled_prototypes, unlabeled_prototypes], 
        batch_key='query'
    )
    adata_latent_full.obs['cell_type_pred'][adata_latent_full.obs['query'].isin(['0'])] = np.nan
    sc.pp.neighbors(adata_latent_full, n_neighbors=300)
    sc.tl.umap(adata_latent_full)

    # Also get adata without prototypes
    adata_no_prototypes = adata_latent_full[adata_latent_full.obs['query'].isin(['0', '1'])]

    # Save these 
    objpath = f"{outdir}/objects"
    if not os.path.exists(objpath):
        os.makedirs(objpath)

    print("~~~~~~~~~~~~~ Saving objects ~~~~~~~~~~~~")
    if 'patient_number' in adata_no_prototypes.obs:
        adata_no_prototypes.obs = adata_no_prototypes.obs.drop('patient_number', axis=1)
        adata_no_prototypes.obs = adata_no_prototypes.obs.drop('label__machine_pred', axis=1)
        adata_no_prototypes.obs['cluster'] = adata_no_prototypes.obs['cluster'].astype(str)


    if 'patient_number' in adata_latent_full.obs:
        adata_latent_full.obs = adata_latent_full.obs.drop('patient_number', axis=1)
        adata_latent_full.obs = adata_latent_full.obs.drop('label__machine_pred', axis=1)
        adata_latent_full.obs['cluster'] = adata_latent_full.obs['cluster'].astype(str)

    columns_to_remove = [col for col in adata_no_prototypes.obs.columns if 'Keras' in col or 'Azimuth' in col or 'Celltypist' in col]
    adata_no_prototypes.obs = adata_no_prototypes.obs.drop(columns=columns_to_remove)
    adata_no_prototypes.obs.drop(['cluster','label__machine_pred'], axis=1, inplace=True)
    adata_no_prototypes.write_h5ad(objpath + "/adata_ref_query_no_prototypes.h5ad")
    columns_to_remove = [col for col in adata_latent_full.obs.columns if 'Keras' in col or 'Azimuth' in col or 'Celltypist' in col]
    adata_latent_full.obs = adata_latent_full.obs.drop(columns=columns_to_remove)
    adata_latent_full.obs.drop(['cluster','label__machine_pred'], axis=1, inplace=True)
    adata_latent_full.write_h5ad(objpath + "/adata_ref_query_full.h5ad")

    # Plot this according to predictions and tissue
    print("~~~~~~~~~~ Plotting results ~~~~~~~~~~~")
    sc.pl.umap(
        adata_no_prototypes, 
        color='cell_type_pred',
        show=False, 
        frameon=False,
        save="_TI-rectum_no_prototypes_label__machine.png"
    )

    adata_no_prototypes.obs['query'] = pd.Categorical(adata_no_prototypes.obs['query'], categories=['0','1'])
    adata_no_prototypes.obs['tissue'] = adata_no_prototypes.obs['query'].map({'0': 'TI', '1': 'rectum'})

    sc.pl.umap(
        adata_no_prototypes, 
        color='tissue',
        show=False, 
        frameon=False,
        save="_TI-rectum_no_prototypes_tissue.png"
    )

    # Inspect the uncertainty
    sc.pl.umap(
        adata_no_prototypes, 
        color='cell_type_uncert',
        show=False, 
        frameon=False,
        cmap='magma',
        vmax=1,
        save="_TI-rectum_no_prototypes_scPoli_annotation_uncertainty.png"
    )
        
    
# Execute
if __name__ == '__main__':
    main()
