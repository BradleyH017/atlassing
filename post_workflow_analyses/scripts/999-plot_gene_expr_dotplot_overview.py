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
    # input_f="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/ASHG_results_round3/combined/objects/adata_PCAd_batched_umap_add_expression.h5ad"
    # cluster_col="leiden"
    inherited_options = parse_options()
    input_f=inherited_options.input_f
    gene_list=inherited_options.gene_list
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
    timarkers = {
        "Epithelial": {
            'Epithelial': ["EPCAM", "CDH1", "KRT19"], # High level markers
            'Stem cells OLFM4+': ['OLFM4'], #,'ASCL2', 'MYC', 'SMOC2' 'Extra markers': ['SELENOP', "MAF", "ALDOB", "FABP6", "IL32", "BCAS1"], 
            'Stem cells OLFM4+ LGR5+': ['LGR5'], 
            'Stem cells MKI67+': ['MKI67'],
            'Enterocytes BEST4+': [ 'BEST4', 'CA7'], 
            'Enterocytes OLFM4+':["OLFM4"], #'EPHB2', 'MYC', 'GPX2',   
            'Enterocytes GPX2': ["GPX2"],
            'Enterocytes KR20T+':["KRT20"],
            'Enterocytes IFI27+':["IFI27"],  
            'Enterocytes ALDOB+':["ALDOB"], 
            'Enterocytes top/middle villus': ['ALPI', 'APOA4', 'APOC3'],                  
            'Goblet cells': ['CLCA1', 'SPDEF', 'FCGBP', 'MUC2', 'ATOH1', 'SPINK4'],   
            'Goblet cells BCAS1+': ['BCAS1'],  
            'Goblet cells MKI67+': ['MKI67'],
            'Goblet cells top/middle villus': ['KLF4', 'NT5E', 'SLC17A5', 'EGFR'],     
            'Enteroendocrine cells': ["NTS","PYY","GCG","PRPH","ISL1"],
            "Enterochromaffin cells":["TPH1", "NPW","CES1"], # From Kong "REG4" #'Paneth cells': ['DEFA5', 'DEFA6', 'REG3A'], #DEFA5, DEFA6, REG3A 
            'Tuft cells': ['POU2F3', 'PTGS1', 'LRMP', 'SH2D6', 'PLCG2'] 
        },
        "Myeloid": {
            'Immune': ["PTPRC"], # High level marker
            'Myeloid': ["ITGAM", "CD14", "CSF1R", "TYROBP"], # Myeloid high level
            'Monocytes S100A8/9+': ['S100A8', 'S100A9'], #'Monocytes SOD2+': ['SOD2'],
            'Monocytes SOD2+': ["SOD2"], 
            'Macrophages CCL3/4+ CXCL9/10+': ["CCL3", "CCL4", "CXCL9", "CXCL10", 'IL1A'], 
            'Macrophages ITGA4+': ["ITGA4"], 
            'Macrophages CD163+': ["CD163",'MAF',"CSF1R","SELENOP", "C1QA", "C1QB", "C1QC"],
            "cDC1":["XCR1", "BATF3", "CLEC9A"], #"IRF8", 
            "pDCs/cDC2":["IL3RA", "IRF4", "ZEB1", "FLT3"],#, "CD207", "CLEC10A" 
            'Mast cells': ["MS4A2", "TPSAB1"]
        },
        "T": {
            'Immune': ["PTPRC"], # high level
            'T': ["CD3D", "CD3E", "CD3G", "CCR7", "IL7R", "TRAC"], # high level
            'T cells CD4+ memory': ['CD4', 'CD40LG', 'IL7R'], 
            'T cells CD4+ memory CXCR6+': ['CXCR6'], 
            'T cells CD4+ naive': ['CCR7', 'SELL'], #'LEF1' 
            'T cells CD8+ LEF1+': ['LEF1'], 
            'T cells CD8+ PASK+': ['PASK'], 
            'T cells CD4+ Treg': ['FOXP3', 'TIGIT', 'CTLA4'], #'BATF', 'ICA1', 'MAF','IL10' 
            'T cells CD4- CD8-': ['CD8A', 'CD8B', "MBNL1"],
            'T cells CD8+ TRGC2+': ['TRGC2'],#'ITGAE', 'ITGA1' 
            'T cells CD8+ GZMK+': ['GZMK', 'GZMH', 'KLRG1'],#, 'NKG7' 
            'T cells gd': ['TRGC1', 'TRDC', 'GZMA', 'GZMB','FCER1G'], #'CD7', #, 'GZMA', 'TRDC', 'FCER1G' 
            'ILCs': ['IL1R1', 'ALDOC', 'LST1', "TNFSF4"] # MIX WITH NK CELLS  ? #RORC, IL1R1, IL23R, KIT, TNFSF4, PCDH9 'NCAM1'
        } ,
        "B": { 
            'Immune': ["PTPRC"], # High level
            'B': ["CD79A", "MS4A1", "CD79B"], # High level
            'B cells': ['FAU','EEF1A1', 'UBA52', 'RPS14'],
            'B cells activated': ['CKS1B', 'RRM2', 'STMN1','TUBA1B', 'HMGN2', 'HMGB2','MKI67'],  
            'B cells naive': ['IGHD', 'IGHM', 'FCER2'],
            'B cells memory FCRL2+': ['BANK1', "FCRL2"],    
            'B cells germinal centre/plasmablasts': ['CD19', 'TCL1A'],  
            'Plasma B cells': ['CD38', "XBP1", 'JCHAIN',]
        },
        "Mesenchymal": { 
            "Mesenchymal": ["COL1A1", "COL1A2", "COL6A2"], # High level
            'Fibroblasts': ['ADAMDEC1', 'PDGFRA', 'BMP4','LUM', 'COL1A2', 'COL3A1', 'COL1A1'],
            'Endothelial cells': ['PECAM1', 'VWF', 'PLVAP', 'ACKR1', 'CD36'], 
            'Pericyte cells': ['NOTCH3', 'PDGFRB', 'CSPG4'] #,'PDGFRB' NOTCH3, MCAM/CD146, RGS5 'HIGD1B', 'STEAP4','FABP4', 
            },
        "BAD": {
            "BAD": ["MALAT1", "XIST", "TSIX"]
        }
        
    }
     
    # Define the outdir params
    sc.settings.figdir="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/ASHG_results_round3/combined/figures/"
    sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")

    # Add category
    adata.obs['cat'] = adata.obs['leiden'].str.split('_').str[0].astype(str)
    cats = np.unique(adata.obs['cat'])
    
    # Plot dotplot
    for c in cats:
        mask = adata.obs['cat'] == c
        for r in timarkers.keys():
            if c in timarkers.keys():
                print(f"~~~~~~~ Plotting {r} markers in {c} ~~~~~~~~~~")
                sc.pl.dotplot(adata[mask,:].copy(), timarkers[r], groupby='leiden',gene_symbols='gene_symbols_unique', dendrogram=False, standard_scale='var', show=True, save=f"_timarkers_{r}_in_{c}.png") 
