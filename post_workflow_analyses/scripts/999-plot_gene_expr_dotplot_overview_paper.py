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
import seaborn as sns
import matplotlib.pyplot as plt

# Load the args
def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                Plot expression of specific genes
                """
        )
    
    parser.add_argument(
            '-if', '--input_f',
            action='store',
            dest='input_f',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-cc', '--cluster_col',
            action='store',
            dest='cluster_col',
            required=True,
            help=''
        )
    
def main():
    # Parse options
    # input_f="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/results_round3/combined/objects/adata_PCAd_batched_umap_add_expression.h5ad"
    # cluster_col="Final_labels"
    inherited_options = parse_options()
    input_f=inherited_options.input_f
    cluster_col=inherited_options.cluster_col
    
    # Load anndata
    adata = sc.read_h5ad(input_f)
    
    # Make sure gene names are unique
adata.var['ENS'] = adata.var_names.copy()
adata.var_names = list(adata.var['gene_symbols'])
adata.var_names_make_unique()
adata.var['gene_symbols_unique'] = adata.var_names.copy()
adata.var.set_index("ENS", inplace=True)

    # Generate dict for previously used marker genes
markers = {
    "Category": {
        "Colonocyte": ["CA2", "SLC26A2", "FABP1"],
        "Enterocyte": ["RBP2", "ANPEP1", "FABP2"],
        "Stem": ["LGR5", "OLFM4", "ASCL2", "SMOC2", "RGMB"],
        "Mesenchymal": ["COL1A1", "COL1A2", "COL6A2"],
        "B": ["CD79A", "MS4A1", "MS4A1", "CD79B"],
        "B cell plasma": ["MZB1", "JCHAIN"],
    },
    "Secretory": {
        "Goblet": ["MUC2", "CLCA1", "FCGBP"], 
        "Tuft": ["POU2F3", "SH2D6", "LRMP", "TRPM5"],
        "Paneth": ["DEFA5", "DEFA6"],
        "Enteroendocrine": ["NTS", "PYY", "GCG", "ISL1"],
        "Enterochromaffin": ["TPS1", "CES1"]
    },
    "Mesenchymal": {
        "Fibroblasts": ["ADAMDEC1", "PDGFRA", "BMP4", "LUM", "COL1A2", "COL3A1", "COL1A1", "C3", "CPM", "NPY", "GRIN2A", "ADAMDEC1"],
        #"Glial": ["NRXY"],
        "Smooth muscle": ["ACTA2"],
        "Pericytes": ["PDGFRB"],
        "Endothelial": ["PECAM1", "VWF", "PLVAP", "ACKR1", "CD36", "TLL1"],
        "Myofibroblasts": ["GREM2", "ADGRL3", "EEF1A1"]
    },
    "Plasma": {
        "Plasma": ["CD38"]
    },
    "B":{ 
        "Activated": ["CKS1B", "RRM2", "STMN1", "MKI67"],
        "Atypical": ["FCRL2", "ITGAX", "TBX21"],
        "Naive": ["IGHD", "FCER2"],
        "Germinal centre/Plasmablast": ["CD19", "TCL1A", "MKI67"],
        "Resting": ["CD69", "CD83"],
        "Memory": ["BANK1", "FCRL2", "CD27", "HIVEP3", "PTPRJ", "PDE4D", "TEX9"]
    },
    "Myeloid": {
        "Monocytes": ["S100A8", "S100A9", "CD14", "FCGR3A", "SOD2", "CXCL9", "CXCL10"],
        "Macrophage": ["ITGA4", "CSF1R", "MAF", "C1QA", "C1QB", "C1QC", "CD163", "MRC1", "SELENOP","HLA-DRA"],
        "Dendritic cells": ["CLEC9A", "CLEC10A", "ZEB1", "IL3RA"]
    }
    #,
   # "Colonocyte": {
   #     "All": ["OLFM4", "MKI67", "BEST4", "CEACAM7", "KRT20", "IFI27", "RBFOX1"]
   # }
}
 
    # Define the outdir params
sc.settings.figdir="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/other_paper_figs"
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")

# Plot dotplot within categories
cats = np.unique(adata.obs['Category'])
cats = np.intersect1d(cats, list(markers.keys()))
for c in cats:
    out=f"{sc.settings.figdir}/dotplot__papermarkers_{c}.png"
    if not os.path.exists(out):
        print(f"Plotting: {c}")
        mask = adata.obs['Clustering_category'] == c
        sc.pl.dotplot(adata[mask,:].copy(), markers[c], groupby='Final_labels',gene_symbols='gene_symbols_unique', dendrogram=False, standard_scale='var', show=True, save=f"_papermarkers_{c}.png") 

    # Also add the other markers (suggestions from Gareth)
add_dict = {
    "Myeloid": {
        "General DC": ["FLT3", "IRF4", "IRF8", "CD1C", "THBD", "HLA-DRA", "BATF3", "SLAMF7"], # Can't find "CD141", this is TBHD,
        "cDC1": ["CLEC9A", "XCR1"],
        "cDC2": ["CLEC10A", "ZEB1"],
        "pDC": ["IL3RA"],
        "Blood monocytes": ["CD14", "FCGR3A"],
        "General Macrophage": ["CSF1R", "MAF", "ITGA4"],
        "Resident Mac": ["C1QA", "C1QB", "C1QC", "CD163", "MRC1", "SELENOP", "HLA-DRA"],
        "Monocytes": ["S100A8", "S100A9", "S100A12", "FCN1", "VCAN", "CD300E", "SOD2"],
        "Intermediate": ["CCL3", "CCL4", "CXCL9", "CXCL10"],
        "For fun": ["IL1B", "IL1A", "ETS2"],
        "For MMP9+ subset": ["MMP9"],
        "Proliferation": ["MKI67"]
    },
    "T": {
        "T cell subsets": ["CD3D", "CD3E", "TRAC", "TRBC1", "CD4", "CD8A", "CD8B", "CD40LG", "THEMIS", "IL7R", "CCR7", "SELL", "LEF1", "PASK", "ITGA1", "CD96", "MBNL1", "TIGIT", "CD28", "FOXP3", "CTLA4", "TRGC1", "TRGC2", "TRDC", "CXCR6", "GZMK"],
        "NK genes": ["FCGR3A", "NCAM1", "KLRG1", "GZMA", "GZMB", "CD160", "SPON2", "CD247", "NKG7", "XCL1", "XCL2", "AREG", "CD44", "EEF1A1"],
        "ILC genes": ["IL1R1", "ALDOC", "LST1", "GATA3", "RORC", "TBX21", "EOMES", "NCR1", "NCR2", "NCR3", "IL23R"],
        "Proliferation": ["MKI67"]
    }
}
    
# plot
for c in add_dict.keys():
    print(f"~~~~~~~ Plotting additional {c} markers~~~~~~~~~~")
    mask = adata.obs['cat'] == c
    sc.pl.dotplot(adata[mask,:].copy(), add_dict[c], groupby='leiden',gene_symbols='gene_symbols_unique', dendrogram=False, standard_scale='var', show=True, save=f"_additional_markers_{c}.png") 

# Also plot the T/NK genes in myeloid
c="Myeloid"
mask = adata.obs['cat'] == c
sc.pl.dotplot(adata[mask,:].copy(), add_dict['T'], groupby='leiden',gene_symbols='gene_symbols_unique', dendrogram=False, standard_scale='var', show=True, save=f"_additional_T_NK_markers_{c}.png") 


# Do with specific subset
mask = adata.obs['leiden'].isin(["Myeloid_8", "Myeloid_9", "Myeloid_11", "Myeloid_16"])
sc.pl.dotplot(adata[mask,:].copy(), add_dict['Myeloid'], groupby='leiden',gene_symbols='gene_symbols_unique', dendrogram=False, standard_scale='var', show=True, save=f"_additional_markers_Myeloid_subset.png") 


# Plot the QC params for a specific cluster in a specific category
adata.obs['log_n_genes_by_counts'] = np.log10(adata.obs['n_genes_by_counts'])
adata.obs['log_total_counts'] = np.log10(adata.obs['total_counts'])
cluster="Epithelial_31"
Category="Secretory"
qc_cols=["pct_counts_gene_group__mito_transcript","log_n_genes_by_counts","log_total_counts"]
mask = adata.obs['Category'] == Category
annots = pd.unique(adata.obs.loc[mask,'leiden'].astype(str))
tds = pd.unique(adata.obs.loc[mask, 'tissue_disease'].astype(str))
tissue_disease_colors = {"r_Healthy": "#E07B28", "blood_CD": "#BB5566", "ti_Healthy": "#117733", "ti_CD": "#808080"}
for qc in qc_cols:
    print(f"Plotting {qc} per cluster - highlighting {cluster}")
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    for a in annots:
        mask_a=adata.obs['leiden'] == a
        data = adata.obs.loc[mask_a,qc]
        if a == cluster:
            sns.distplot(data, hist=False, rug=True, label=a, kde_kws={'color': 'blue'})
        else:
            sns.distplot(data, hist=False, rug=True, label=a, kde_kws={'color': 'orange'})
    #
    plt.legend()
    plt.xlabel(qc)
    plt.title(f"Distribution of {qc} - {Category}")
    plt.savefig(f"other_paper_figs/{qc}_per_cluster_highlight_{cluster}_in_{Category}.png", bbox_inches='tight')
    plt.clf()
    # Also colour by tissue x disease status
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    for a in annots:
        mask_a=adata.obs['leiden'] == a
        data = adata.obs.loc[mask_a,qc]
        tissue_disease_max = adata.obs.loc[mask_a, 'tissue_disease'].value_counts().idxmax()
        color = tissue_disease_colors.get(tissue_disease_max, "black")
        sns.distplot(data, hist=False, rug=True, label=a, kde_kws={'color': color})
    #
    plt.legend()
    plt.xlabel(qc)
    plt.title(f"Distribution of {qc} - {Category}")
    plt.savefig(f"/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/ASHG_results_round3/combined/figures/{qc}_per_cluster_highlight_tissue_disease_in_{Category}.png", bbox_inches='tight')
    plt.clf()
    
# Get medians per cluster erc
rows = []
for leiden_value in adata.obs.loc[mask,'leiden'].astype(str).unique():
    print(leiden_value)
    mask_a = adata.obs['leiden'] == leiden_value
    # Calculate medians for the specified columns
    median_pct_counts_mito_transcript = adata.obs.loc[mask_a, 'pct_counts_gene_group__mito_transcript'].median()
    median_log_n_genes_by_counts = adata.obs.loc[mask_a, 'log_n_genes_by_counts'].median()
    median_log_total_counts = adata.obs.loc[mask_a, 'log_total_counts'].median()
    most_common_tissue_disease = adata.obs.loc[mask_a, 'tissue_disease'].value_counts().idxmax()
    rows.append({
        'leiden': leiden_value,
        'median_pct_counts_mito_transcript': median_pct_counts_mito_transcript,
        'median_log_n_genes_by_counts': median_log_n_genes_by_counts,
        'median_log_total_counts': median_log_total_counts,
        'most_common_tissue_disease': most_common_tissue_disease
    })

# Create a DataFrame from the list of rows
summary_df = pd.DataFrame(rows)