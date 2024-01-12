#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-01-12'
__version__ = '0.0.1'

# Change dir
import os
cwd = os.getcwd()
print(cwd)
os.chdir("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results")
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
print("Loaded libraries")

# Parse script options (ref, query, outdir)
def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                QC of tissues together
                """
        )
    
    parser.add_argument(
            '-b', '--blood_file',
            action='store',
            dest='blood_file',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-t', '--TI_file',
            action='store',
            dest='TI_file',
            required=True,
            help=''
        )

    parser.add_argument(
            '-r', '--rectum_file',
            action='store',
            dest='rectum_file',
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
            '-d', '--discard_other_inflams',
            action='store',
            dest='discard_other_inflams',
            required=True,
            help=''
        )

    parser.add_argument(
        '-nUMI', '--min_nUMI',
        action='store',
        dest='min_nUMI',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-abs_nUMI', '--use_absolute_nUMI',
        action='store',
        dest='use_absolute_nUMI',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-rel_nMAD_thresh', '--relative_nMAD_threshold',
        action='store',
        dest='relative_nMAD_threshold',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-nGene', '--min_nGene',
        action='store',
        dest='min_nGene',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-abs_nGene', '--use_absolute_nGene',
        action='store',
        dest='use_absolute_nGene',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-MTgut', '--MT_thresh_gut',
        action='store',
        dest='MT_thresh_gut',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-MTblood', '--MT_thresh_blood',
        action='store',
        dest='MT_thresh_blood',
        required=True,
        help=''
    )

    parser.add_argument(
        '-abs_MT', '--use_absolute_MT',
        action='store',
        dest='use_absolute_MT',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-samp_nGene_blood', '--min_mean_nGene_per_samp_blood',
        action='store',
        dest='min_mean_nGene_per_samp_blood',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-samp_nCount_gut', '--min_mean_nCount_per_samp_gut',
        action='store',
        dest='min_mean_nCount_per_samp_gut',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-samp_nCount_gut', '--min_mean_nCount_per_samp_gut',
        action='store',
        dest='min_mean_nCount_per_samp_gut',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-samp_nGene_blood', '--min_mean_nGene_per_samp_blood',
        action='store',
        dest='min_mean_nGene_per_samp_blood',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-samp_nGene_gut', '--min_mean_nGene_per_samp_gut',
        action='store',
        dest='min_mean_nGene_per_samp_gut',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-use_abs_per_samp', '--use_abs_per_samp',
        action='store',
        dest='use_abs_per_samp',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-filt_blood_keras', '--filt_blood_keras',
        action='store',
        dest='filt_blood_keras',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-n_var', '--n_variable_genes',
        action='store',
        dest='n_variable_genes',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-rpg', '--remove_problem_genes',
        action='store',
        dest='remove_problem_genes',
        required=True,
        help=''
    )

    return parser.parse_args()


def main():
    # Parse options
    inherited_options = parse_options()
    blood_file = inherited_options.blood_file
    # blood_file = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/yascp_analysis/blood/results/merged_h5ad/outlier_filtered_adata.h5ad"
    TI_file = inherited_options.TI_file
    # TI_file = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/anderson_ti_freeze003_004-otar-processed.h5ad"
    rectum_file = inherited_options.rectum_file
    # rectum_file = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/2023_09_rectum/adata.h5ad"
    outdir = inherited_options.outdir
    # outdir = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/tissues_combined"
    discard_other_inflams = inherited_options.discard_other_inflams
    # discard_other_inflams = "yes"
    min_nUMI = inherited_options.min_nUMI.astype('float')
    # min_nUMI = 200
    use_absolute_nUMI = inherited_options.use_absolute_nUMI
    # use_absolute_nUMI = "yes"
    relative_nMAD_threshold = inherited_options.relative_nMAD_threshold.astype('float')
    # relative_nMAD_threshold = 2.5
    min_nGene = inherited_options.min_nGene.astype('float')
    # min_nGene = 400
    use_absolute_nGene = inherited_options.use_absolute_nGene
    # use_absolute_nGene = "yes"
    MTgut = inherited_options.MTgut.astype('float')
    # MTgut = 50
    MTblood = inherited_options.MTblood.astype('float')
    # MTblood = 20
    use_absolute_MT = inherited_options.use_absolute_MT
    # use_absolute_MT = "yes"
    min_mean_nCount_per_samp_blood = inherited_options.min_mean_nCount_per_samp_blood.astype('float')
    # min_mean_nCount_per_samp_blood = 3000
    min_mean_nCount_per_samp_gut = inherited_options.min_mean_nCount_per_samp_gut.astype('float')
    # min_mean_nCount_per_samp_gut = 7500
    min_mean_nGene_per_samp_blood = inherited_options.min_mean_nGene_per_samp_blood.astype('float')
    # min_mean_nGene_per_samp_blood = 1000
    min_mean_nGene_per_samp_gut = inherited_options.min_mean_nGene_per_samp_gut.astype('float')
    # min_mean_nGene_per_samp_gut = 1750
    use_abs_per_samp = inherited_options.use_abs_per_samp
    # use_abs_per_samp = "yes"
    filt_blood_keras = inherited_options.filt_blood_keras
    # filt_blood_keras = "no"
    n_variable_genes = inherited_options.n_variable_genes.astype('float')
    # n_variable_genes = 4000
    remove_problem_genes = inherited_options.remove_problem_genes
    # remove_problem_genes = "yes"

    # Define output directories
    figpath = f"{outdir}/figures"
    if os.path.exists(figpath) == False:
        os.mkdir(figpath)

    tabpath = f"{outdir}/tables"
    if os.path.exists(tabpath) == False:
        os.mkdir(tabpath)

    objpath = f"{outdir}/objects"
    if os.path.exists(objpath) == False:
        os.mkdir(objpath)

    # Define global figure directory
    sc.settings.figdir=figpath
    sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")

    # Load the given files and perform initial formatting / filters
    blood = sc.read_h5ad(blood_file)
    blood.obs['tissue'] = "blood"
    blood.obs = blood.obs.rename(columns = {'Keras:predicted_celltype_probability': 'label__machine_probability'})
    blood.obs = blood.obs.rename(columns = {'Keras:predicted_celltype': 'label__machine'})
    gene_conv = blood.var[['gene_symbols']]
    gene_conv = gene_conv.reset_index()
    TI = sc.read_h5ad(TI_file)
    TI.obs['tissue'] = "TI"
    TI.obs = TI.obs.rename(columns = {'predicted_celltype_probability': 'label__machine_probability'})
    rectum = sc.read_h5ad(rectum_file)
    rectum.obs['tissue'] = "rectum"
    rectum.obs = rectum.obs.rename(columns = {'Keras:predicted_celltype_probability': 'label__machine_probability'})
    rectum.obs = rectum.obs.rename(columns = {'Keras:predicted_celltype': 'label__machine'})
    adata = ad.concat([blood, TI, rectum])
    adata.var = adata.var.reset_index()
    adata.var = adata.var.merge(gene_conv, on="index", how="left")
    adata.var.set_index("index", inplace=True)
    del blood, TI, rectum
    print("Loaded the input files and merged")

    # Discard other inflams if wanted to 
    if discard_other_inflams == "yes":
        adata = adata[adata.obs['disease_status'].isin(['healthy', 'cd'])]

    # Append the annotation (category) and lineage
    annot = pd.read_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/highQC_TI_discovery/data-clean_annotation_full.csv")
    annot['label__machine'] = annot['label__machine_retired']
    adata.obs['cell'] = adata.obs.index
    adata.obs = adata.obs.merge(annot[['category__machine', 'label__machine']], on='label__machine', how ='left')
    adata.obs.set_index('cell', inplace=True)
    adata.obs['lineage'] = np.where(adata.obs['category__machine'].isin(['B_Cell', 'B_Cell_plasma', 'T_Cell', 'Myeloid']), 'Immune', "")
    adata.obs['lineage'] = np.where(adata.obs['category__machine'].isin(['Stem_cells', 'Secretory', 'Enterocyte']), 'Epithelial', adata.obs['lineage'])
    adata.obs['lineage'] = np.where(adata.obs['category__machine']== 'Mesenchymal', 'Mesenchymal', adata.obs['lineage'])

    ####################################
    ######### Preliminary QC ###########
    ####################################
    print("~~~~~~~~~~~~ Conducting preliminary QC ~~~~~~~~~~~~~")
    qc_path = f"{figpath}/QC"
    if os.path.exists(qc_path) == False:
        os.mkdir(qc_path)

    # 1. Discard cells with very low nUMI
    print("Filtration of cells with very low nUMI")
    adata.obs[["total_counts"]] = adata.obs[["total_counts"]].astype('float')
    print(f"The minimum nUMI is: {str(min(adata.obs.total_counts))}")
    # plot a distribution of nUMI per tissue
    adata.obs['total_counts'] = adata.obs['total_counts'].astype(int)
    tissues = np.unique(adata.obs['tissue'])
    cats = np.unique(adata.obs['category__machine'])
    lins = np.unique(adata.obs['lineage'])

    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    for t in tissues:
        data = np.log10(adata.obs[adata.obs.tissue == t].total_counts)
        sns.distplot(data, hist=False, rug=True, label=t)

    plt.legend()
    plt.xlabel('log10(nUMI)')
    plt.axvline(x = np.log10(min_nGene), color = 'red', linestyle = '--', alpha = 0.5)
    plt.savefig(qc_path + '/raw_nUMI_per_tissue.png', bbox_inches='tight')
    plt.clf()

    # Plot per category, within tissue
    for t in tissues:
        print(t)
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
        for c in cats:
            data = np.log10(adata.obs[(adata.obs.tissue == t) & (adata.obs.category__machine == c)].total_counts)
            sns.distplot(data, hist=False, rug=True, label=c)
        
        plt.legend()
        plt.xlabel('log10(nUMI)')
        plt.axvline(x = np.log10(min_nGene), color = 'red', linestyle = '--', alpha = 0.5)
        plt.title(t)
        plt.savefig(f"{qc_path}/raw_nUMI_per_category_{t}.png", bbox_inches='tight')
        plt.clf()

    # Plot per lineage, within tissue
    for t in tissues:
        print(t)
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
        for l in lins:
            data = np.log10(adata.obs[(adata.obs.tissue == t) & (adata.obs.lineage == l)].total_counts)
            sns.distplot(data, hist=False, rug=True, label=l)
        
        plt.legend()
        plt.xlabel('log10(nUMI)')
        plt.axvline(x = np.log10(min_nGene), color = 'red', linestyle = '--', alpha = 0.5)
        plt.title(t)
        plt.savefig(f"{qc_path}/raw_nUMI_per_lineage_{t}.png", bbox_inches='tight')
        plt.clf()

    # Filter for this cut off if using an absolute cut off, or define a relative cut off
    if use_absolute_nUMI == "yes":
        adata = adata[adata.obs['total_counts'] > min_nUMI]
    else:
        # TO DO: Write in the relative nUMI filtration
        print("Defining nMAD cut off")

    # 2. nGene
    print("Filtration of cells with low nGenes")
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    for t in tissues:
        data = np.log10(adata.obs[adata.obs.tissue == t].n_genes_by_counts)
        sns.distplot(data, hist=False, rug=True, label=t)

    plt.legend()
    plt.xlabel('log10(nGene)')
    plt.axvline(x = np.log10(min_nGene), color = 'red', linestyle = '--', alpha = 0.5)
    plt.savefig(qc_path + '/nUMI_filt_nGene_per_tissue.png', bbox_inches='tight')
    plt.clf()

    # Plot per category, within tissue
    for t in tissues:
        print(t)
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
        for c in cats:
            data = np.log10(adata.obs[(adata.obs.tissue == t) & (adata.obs.category__machine == c)].n_genes_by_counts)
            sns.distplot(data, hist=False, rug=True, label=c)
        
        plt.legend()
        plt.xlabel('log10(nGene)')
        plt.axvline(x = np.log10(min_nGene), color = 'red', linestyle = '--', alpha = 0.5)
        plt.title(t)
        plt.savefig(f"{qc_path}/nUMI_filt_nGene_per_category_{t}.png", bbox_inches='tight')
        plt.clf()

    # Plot per lineage, within tissue
    for t in tissues:
        print(t)
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
        for l in lins:
            data = np.log10(adata.obs[(adata.obs.tissue == t) & (adata.obs.lineage == l)].n_genes_by_counts)
            sns.distplot(data, hist=False, rug=True, label=l)
        
        plt.legend()
        plt.xlabel('log10(nGene)')
        plt.axvline(x = np.log10(min_nGene), color = 'red', linestyle = '--', alpha = 0.5)
        plt.title(t)
        plt.savefig(f"{qc_path}/nUMI_filt_nGene__per_lineage_{t}.png", bbox_inches='tight')
        plt.clf()
        

    # Filter for this cut off if using an absolute cut off, or define a relative cut off
    if use_absolute_nGene == "yes":
        adata = adata[adata.obs['n_genes_by_counts'].astype(int) > min_nGene]
    else:
        # TO DO: Write in the relative nUMI filtration
        print("Defining nMAD cut off")
        
    # 3. MT% 
    print("Filtration of cells with abnormal/low MT%")
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    for t in tissues:
        data = adata.obs[adata.obs.tissue == t].pct_counts_gene_group__mito_transcript
        sns.distplot(data, hist=False, rug=True, label=t)

    plt.legend()
    plt.xlabel('MT%')
    plt.axvline(x = MTgut, color = 'green', linestyle = '--', alpha = 0.5)
    plt.axvline(x = MTblood, color = 'red', linestyle = '--', alpha = 0.5)
    plt.savefig(qc_path + '/nUMI_nGene_filt_MT_per_tissue.png', bbox_inches='tight')
    plt.clf()

    # Plot per category, within tissue
    for t in tissues:
        print(t)
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
        for c in cats:
            data = adata.obs[(adata.obs.tissue == t) & (adata.obs.category__machine == c)].pct_counts_gene_group__mito_transcript
            sns.distplot(data, hist=False, rug=True, label=c)
        
        plt.legend()
        plt.xlabel('MT%')
        if t == "blood":
            plt.axvline(x = MTblood, color = 'red', linestyle = '--', alpha = 0.5)
        else:
            plt.axvline(x = MTgut, color = 'green', linestyle = '--', alpha = 0.5)
        
        plt.title(t)
        plt.savefig(f"{qc_path}/nUMI_nGene_filt_MT_per_category_{t}.png", bbox_inches='tight')
        plt.clf()

    # Plot per lineage, within tissue
    for t in tissues:
        print(t)
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
        for l in lins:
            data = np.log10(adata.obs[(adata.obs.tissue == t) & (adata.obs.lineage == l)].n_genes_by_counts)
            sns.distplot(data, hist=False, rug=True, label=l)
        
        plt.legend()
        plt.xlabel('log10(nGene)')
        if t == "blood":
            plt.axvline(x = MTblood, color = 'red', linestyle = '--', alpha = 0.5)
        else:
            plt.axvline(x = MTgut, color = 'green', linestyle = '--', alpha = 0.5)
        
        plt.title(t)
        plt.savefig(f"{qc_path}/nUMI_filt_nGene__per_lineage_{t}.png", bbox_inches='tight')
        plt.clf()

    # Filter on the basis of these thresholds
    if use_absolute_MT == "yes":
        blood_mask = (adata.obs['tissue'] == 'blood') & (adata.obs['pct_counts_gene_group__mito_transcript'] < MTblood)
        gut_tissue_mask = (adata.obs['tissue'] != 'blood') & (adata.obs['pct_counts_gene_group__mito_transcript'] < MTgut)
        adata = adata[blood_mask | gut_tissue_mask]
    else:
        print("Defining nMAD cut off")
        
    # 4. Remove samples with outlying sequencing depth
    samp_data = np.unique(adata.obs.experiment_id, return_counts=True)
    cells_sample = pd.DataFrame({'sample': samp_data[0], 'Ncells':samp_data[1]})
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    sns.distplot(cells_sample.Ncells)
    plt.xlabel('Cells/sample')
    plt.axvline(x = 500, color = 'red', linestyle = '--', alpha = 0.5)
    ax.set(xlim=(0, max(cells_sample.Ncells)))
    plt.savefig(f"{qc_path}/postQC_cells_per_sample_all.png", bbox_inches='tight')
    plt.clf()

    # Also do this within tissue
    for t in tissues:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
        sns.distplot(cells_sample[cells_sample['sample'].isin(adata.obs[adata.obs['tissue'] == t]['experiment_id'])].Ncells)
        plt.xlabel('Cells/sample')
        plt.axvline(x = 500, color = 'red', linestyle = '--', alpha = 0.5)
        ax.set(xlim=(0, max(cells_sample.Ncells)))
        plt.savefig(f"{qc_path}/postQC_cells_per_sample_{t}.png", bbox_inches='tight')
        plt.clf()
        
    # Have a look at those cells with high number of cells within each tissue
    high_samps = np.array(cells_sample.loc[cells_sample.Ncells > 10000, "sample"])
    low_samps = np.array(cells_sample.loc[cells_sample.Ncells < 500, "sample"])
    # Summarise
    depth_count = pd.DataFrame(index = np.unique(adata.obs.experiment_id), columns=["Mean_nCounts", "nCells", "High_cell_sample", "n_genes_by_counts"])
    for s in range(0, depth_count.shape[0]):
        samp = depth_count.index[s]
        depth_count.iloc[s,1] = adata.obs[adata.obs.experiment_id == samp].shape[0]
        depth_count.iloc[s,0] = sum(adata.obs[adata.obs.experiment_id == samp].total_counts)/depth_count.iloc[s,1]
        depth_count.iloc[s,3] = sum(adata.obs[adata.obs.experiment_id == samp].n_genes_by_counts)/depth_count.iloc[s,1]
        if samp in high_samps:
            depth_count.iloc[s,2] = "Red"
        else: 
            depth_count.iloc[s,2] = "Navy"
        # Also annotate the samples with low number of cells - Are these sequenced very deeply?
        if samp in low_samps:
            depth_count.iloc[s,2] = "Green"

    depth_count["log10_Mean_Counts"] = np.log10(np.array(depth_count["Mean_nCounts"].values, dtype = "float"))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(depth_count["Mean_nCounts"], depth_count["n_genes_by_counts"],  c=depth_count["High_cell_sample"], alpha=0.7)
    plt.xlabel('Mean counts / cell')
    plt.ylabel('Mean genes detected / cell')
    plt.savefig(f"{qc_path}/postQC_mean_counts_ngenes_all.png", bbox_inches='tight')
    plt.clf()

    # Plot the distribution per tissue
    for t in tissues:
        samps = adata.obs[adata.obs['tissue'] == t]['experiment_id']
        data = depth_count[depth_count.index.isin(samps)]["Mean_nCounts"]
        sns.distplot(data, hist=False, rug=True, label=t)

    plt.legend()
    plt.xlabel('Mean nCounts/cell')
    plt.savefig(qc_path + '/postQC_dist_mean_counts.png', bbox_inches='tight')
    plt.clf()

    for t in tissues:
        samps = adata.obs[adata.obs['tissue'] == t]['experiment_id']
        data = depth_count[depth_count.index.isin(samps)]["n_genes_by_counts"]
        sns.distplot(data, hist=False, rug=True, label=t)

    plt.legend()
    plt.xlabel('Mean nCounts/cell')
    plt.savefig(qc_path + '/postQC_dist_mean_nGenes.png', bbox_inches='tight')
    plt.clf()

    # Within tissue
    for t in tissues:
        samps = adata.obs[adata.obs['tissue'] == t]['experiment_id']
        plt.figure(figsize=(8, 6))
        plt.scatter(depth_count[depth_count.index.isin(samps)]["Mean_nCounts"], depth_count[depth_count.index.isin(samps)]["n_genes_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["High_cell_sample"], alpha=0.7)
        plt.xlabel('Mean counts / cell')
        plt.ylabel('Mean genes detected / cell')
        if t == "blood":
            plt.axvline(x = min_mean_nCount_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
            plt.axhline(y = min_mean_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
        else:
            plt.axvline(x = min_mean_nCount_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
            plt.axhline(y = min_mean_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
        
        plt.title(t)
        plt.savefig(f"{qc_path}/postQC_mean_counts_ngenes_{t}.png", bbox_inches='tight')
        plt.clf()

    # These are clearly different across tissues
    # Either apply absolute thresholds or a relative cut off
    if use_abs_per_samp == "yes":
        blood_keep = (depth_count['Mean_nCounts'] > min_mean_nCount_per_samp_blood) & (depth_count['n_genes_by_counts'] > min_mean_nGene_per_samp_blood)
        blood_keep = blood_keep & depth_count.index.isin(adata.obs[adata.obs['tissue'] == "blood"]['experiment_id'])
        blood_keep = blood_keep[blood_keep == True].index
        gut_keep = (depth_count['Mean_nCounts'] > min_mean_nCount_per_samp_gut) & (depth_count['n_genes_by_counts'] > min_mean_nGene_per_samp_gut)
        gut_keep = gut_keep & depth_count.index.isin(adata.obs[adata.obs['tissue'] != "blood"]['experiment_id'])
        gut_keep = gut_keep[gut_keep == True].index
        both_keep = list(np.concatenate([blood_keep, gut_keep]))
        adata = adata[adata.obs['experiment_id'].isin(both_keep)]
    else:
        # Calculating relative threshold
        print("Calculating relative threshold")


    # 5. Final bit of cell/sample QC: see what blood cell categories are left and the strength in their annotations.
    # There should be no epithelial or mesenchymal cells in the blood data! -However this may be an issue with keras
    blood = adata[adata.obs['tissue'] == "blood"]
    # Plot the distribution of the cell confidences across lineages and categories in blood cells
    for c in cats:
        data = blood.obs[blood.obs['category__machine'] == c]['label__machine_probability']
        sns.distplot(data, hist=False, rug=True, label=c)

    plt.legend()
    plt.title("Blood")
    plt.xlabel('Keras annotation probability')
    plt.savefig(qc_path + '/postQC_dist_keras_conf_blood_category.png', bbox_inches='tight')
    plt.clf()

    for c in lins:
        data = blood.obs[blood.obs['lineage'] == c]['label__machine_probability']
        sns.distplot(data, hist=False, rug=True, label=c)

    plt.legend()
    plt.title("Blood")
    plt.xlabel('Keras annotation probability')
    plt.savefig(qc_path + '/postQC_dist_keras_conf_blood_lineage.png', bbox_inches='tight')
    plt.clf()

    if filt_blood_keras == "yes":
        adata = adata[~((adata.obs['tissue'] == 'blood') & (adata.obs['label__machine_probability'].astype('float') < 0.5))]

    ####################################
    #### Expression Normalisation ######
    ####################################
    print("~~~~~~~~~~~~ Conducting expression normalisation ~~~~~~~~~~~~~")
    sc.pp.filter_genes(adata, min_cells=5)

    # Keep a copy of the raw counts
    adata.layers['counts'] = adata.X.copy()

    # Calulate the CP10K expression
    sc.pp.normalize_total(adata, target_sum=1e4)

    # Now normalise the data to identify highly variable genes (Same as in Tobi and Monika's paper)
    sc.pp.log1p(adata)

    # identify highly variable genes and scale these ahead of PCA
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_variable_genes, batch_key='experiment_id')

    # Check for intersection of IG, MT and RP genes in the HVGs
    print("IG")
    print(np.unique(adata.var[adata.var.gene_symbols.str.contains("IG[HKL]V|IG[KL]J|IG[KL]C|IGH[ADEGM]")].highly_variable, return_counts=True))
    print("MT")
    print(np.unique(adata.var[adata.var.gene_symbols.str.contains("^MT-")].highly_variable, return_counts=True))
    print("RP")
    print(np.unique(adata.var[adata.var.gene_symbols.str.contains("^RP")].highly_variable, return_counts=True))

    # Remove these from the highly variable set if desired
    if remove_problem_genes == "yes":
        condition = adata.var['gene_symbols'].str.contains('IG[HKL]V|IG[KL]J|IG[KL]C|IGH[ADEGM]|^MT-|^RP', regex=True)
        adata.var.loc[condition, 'highly_variable'] = False

    # Scale
    sc.pp.scale(adata, max_value=10)

    # Print the final shape of the high QC data
    print(f"The final shape of the high QC data is: {adata.shape}")
    for t in tissues:
        ncells = adata.obs[adata.obs['tissue'] == t].shape[0]
        print(f"From {t}: {ncells}")


    ####################################
    ######### PCA calculation ##########
    ####################################

    # Run PCA
    sc.tl.pca(adata, svd_solver='arpack')

    # Plot PCA
    sc.pl.pca(adata, save="True")

    # PLot Elbow plot
    sc.pl.pca_variance_ratio(adata, log=True, save=True, n_pcs = 50)

    #  Determine the optimimum number of PCs
    # Extract PCs
    pca_variance=pd.DataFrame({'x':list(range(1, 51, 1)), 'y':adata.uns['pca']['variance']})
    # Identify 'knee'
    knee=kd.KneeLocator(x=list(range(1, 51, 1)), y=adata.uns['pca']['variance'], curve="convex", direction = "decreasing")
    knee_point = knee.knee
    elbow_point = knee.elbow
    print('Knee: ', knee_point) 
    print('Elbow: ', elbow_point)
    # Use the 'knee' + 5 additional PCs
    nPCs = knee_point + 5
    print("The number of PCs used for this analysis is {}".format(nPCs))

    # Save the 'knee'
    from numpy import asarray
    from numpy import savetxt
    savetxt(tabpath + "/knee.txt", asarray([[knee_point]]), delimiter='\t')

    # Save the PCA loadings
    loadings = pd.DataFrame(adata.varm['PCs'])
    loadings = loadings.iloc[:,0:knee_point]
    loadings.index = np.array(adata.var.index)
    pcs = list(range(1,knee_point+1))
    cols = ["{}{}".format("PC", i) for i in pcs]
    loadings.columns = cols
    loadings.to_csv(tabdir + "/PCA_loadings.csv")

    # Save the PCA matrix 
    pca = pd.DataFrame(adata.obsm['X_pca'])
    pca = pca.iloc[:,0:knee_point]
    pca.index = adata.obs.index
    pca.to_csv(tabpath + "/PCA_up_to_knee.csv")

    # Save object ahead of batch correction
    adata.write(objpath + "/adata_PCAd.h5ad")

    ####################################
    ######### Batch correction #########
    ####################################
    # 1. scVI
    print("~~~~~~~~~~~~~~~~~~~ Batch correcting with scVI - optimum params ~~~~~~~~~~~~~~~~~~~")
    #Trainer(accelerator="cuda")
    scvi.settings.dl_pin_memory_gpu_training =  True
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="experiment_id")
    model = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
    model.train(use_gpu=True)
    SCVI_LATENT_KEY = "X_scVI"
    adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

    # 2. scVI - default_metrics
    print("~~~~~~~~~~~~~~~~~~~ Batch correcting with scVI - Default params ~~~~~~~~~~~~~~~~~~~")
    model_default = scvi.model.SCVI(adata,  n_latent=30)
    model_default.train(use_gpu=True)
    SCVI_LATENT_KEY_DEFAULT = "X_scVI_default"
    adata.obsm[SCVI_LATENT_KEY_DEFAULT] = model_default.get_latent_representation()


    # 3. scANVI
    # Performing this across lineage
    print("~~~~~~~~~~~~~~~~~~~ Batch correcting with scANVI ~~~~~~~~~~~~~~~~~~~")
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        model,
        adata=adata,
        labels_key="lineage",
        unlabeled_category="Unknown",
    )
    scanvi_model.train(max_epochs=20, n_samples_per_label=100, use_gpu=True)
    SCANVI_LATENT_KEY = "X_scANVI"
    adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)

    # Save pre-benchmark (& Harmony) - may be causing issues
    if os.path.exists(objpath) == False:
        os.mkdir(objpath)

    adata.write(objpath + "/adata_PCAd_batched.h5ad")

    # 4. Harmony
    print("~~~~~~~~~~~~~~~~~~~ Batch correcting with Harmony ~~~~~~~~~~~~~~~~~~~")
    sc.external.pp.harmony_integrate(adata, 'experiment_id', basis='X_pca', adjusted_basis='X_pca_harmony')

    # Save pre-benchmark
    if os.path.exists(objpath) == False:
        os.mkdir(objpath)

    adata.write(objpath + "/adata_PCAd_batched.h5ad")

    # Compute NN and UMAP using the recommended number of PCs
    # Would like to calculate the knee from the latent SCANVI factors, but not sure where this is stored or how to access
    # NOTE: Probably want to be doing a sweep of the NN parameter as have done for all of the data together
    # Using non-corrected matrix
    # Compute UMAP (Will also want to do a sweep of min_dist and spread parameters here) - Do this based on all embeddings
    colby = ["experiment_id", "category__machine", "label__machine"]
    latents = ["X_pca", SCVI_LATENT_KEY, SCANVI_LATENT_KEY, 'X_pca_harmony', SCVI_LATENT_KEY_DEFAULT]
    for l in latents:
        print("Performing on UMAP embedding on {}".format(l))
        # Calculate NN (using 350)
        print("Calculating neighbours")
        sc.pp.neighbors(adata, n_neighbors=350, n_pcs=nPCs, use_rep=l, key_added=l + "_nn")
        # Perform UMAP embedding
        sc.tl.umap(adata, neighbors_key=l + "_nn", min_dist=0.5, spread=0.5)
        adata.obsm["UMAP_" + l] = adata.obsm["X_umap"]
        # Save a plot
        for c in colby:
            if c != "experiment_id":
                sc.pl.umap(adata, color = c, save="_" + l + "_NN" + c + ".png")
            else:
                sc.pl.umap(adata, color = c, save="_" + l + "_NN" + c + ".png", palette=list(mp.colors.CSS4_COLORS.values()))
        # Overwite file after each 
        adata.write(objpath + "/adata_PCAd_batched.h5ad")

    ####################################
    #### Benchmarking integration ######
    ####################################
    bm = Benchmarker(
        adata,
        batch_key="experiment_id",
        label_key="label",
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
    
