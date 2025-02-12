#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-10-3'
__version__ = '0.0.1'


# Change dir
import os
cwd = os.getcwd()
print(cwd)

# Load packages
# module load HGI/softpack/users/eh19/test-single-cell/20
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import argparse

# Load the args
def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                Plot expression of specific genes
                """
        )
    
    parser.add_argument(
            '-ifu', '--input_file_umap',
            action='store',
            dest='input_file_umap',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-ife', '--input_file_expr',
            action='store',
            dest='input_file_expr',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-ifc', '--input_file_combined',
            action='store',
            dest='input_file_combined',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-m', '--matrix',
            action='store',
            dest='matrix',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-gn', '--gene_name',
            action='store',
            dest='gene_name',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-gs', '--gene_symbol',
            action='store',
            dest='gene_symbol',
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
    
    return parser.parse_args()

def main():
    # Parse options
    inherited_options = parse_options()
    input_file = inherited_options.input_file
    matrix = inherited_options.matrix
    gene_name = inherited_options.gene_name
    gene_name = gene_name.split(",")
    gene_symbol = inherited_options.gene_symbol
    gene_symbol = gene_symbol.split(",")
    outdir = inherited_options.outdir
    
    # Testing
    # input_file_umap="results_round3/combined/objects/adata_PCAd_batched_umap.h5ad"
    # input_file_expr="input_v7/adata_raw_input_all.h5ad"
    # input_file_combined = "results_round3/combined/objects/adata_PCAd_batched_umap_add_expression.h5ad"
    # matrix="log1p_cp10k"
    # gene_name=["ENSG00000085978","ENSG00000124203", "ENSG00000172575"]
    # gene_symbol=["ATG16L1", "ZNF831", "RASGPR1"]
    # outdir="results_round3/combined/figures/UMAP"
    
    # Combine the umap and expression file if not already
    if os.path.exists(input_file_combined):
        print("Combining umap and expression")    
        # Load input (backed)
        adata = sc.read_h5ad(input_file_umap, backed="r")
        umap = pd.DataFrame(adata.obsm['X_umap'])
        umap.index = adata.obs.index
        #add_expr = adata.layers[matrix].copy()
        clusters = adata.obs['leiden']
        del adata
        # Now load the raw
        adata = sc.read_h5ad(input_file_expr)
        adata = adata[adata.obs.index.isin(umap.index)]
        all(adata.obs.index == umap.index)
        adata.obsm['X_umap'] = umap.astype(str).copy()
        adata.obsm = {str(k): v for k, v in adata.obsm.items()}
        adata.obsm['X_umap'].columns = ["UMAP1", "UMAP2"]
        #adata.layers[matrix] = add_expr.copy()
        adata.obs['leiden'] = clusters
        adata.write_h5ad(input_file_combined)
    else:
        adata = sc.read_h5ad(input_file_combined, backed="r")
    
    # Define outf
    sc.settings.figdir=outdir
    sc.settings.verbosity = 3
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")
    
    # Plot
    for i, g in enumerate(gene_name):
        outf = f"{gene_symbol[i]}_{matrix}.png"
        sc.pl.umap(adata, color=gene_name[i], legend_loc="on data", frameon=False, title=g, legend_fontoutline=1, legend_fontsize=10, save=outf, show=False)


    