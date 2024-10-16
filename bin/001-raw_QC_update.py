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
import holoviews as hv
from sklearn.decomposition import PCA
hv.extension('matplotlib')
print("Loaded libraries")
import sys
sys.path.append('bin')
import nmad_qc # Import custom functions

# Parse script options (ref, query, outdir)
def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                QC of tissues together
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
            '-d', '--discard_other_inflams',
            action='store',
            dest='discard_other_inflams',
            required=True,
            help=''
        )

    parser.add_argument(
            '-abi', '--all_blood_immune',
            action='store',
            dest='all_blood_immune',
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
        '-nmad_dir', '--nMad_directionality',
        action='store',
        dest='nMad_directionality',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-TM', '--threshold_method',
        action='store',
        dest='threshold_method',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-groups', '--relative_grouping',
        action='store',
        dest='relative_grouping',
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
        '-blood_MT_gut', '--use_MT_thresh_blood_gut_immune',
        action='store',
        dest='use_MT_thresh_blood_gut_immune',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-samp_nCount_blood', '--min_median_nCount_per_samp_blood',
        action='store',
        dest='min_median_nCount_per_samp_blood',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-samp_nCount_gut', '--min_median_nCount_per_samp_gut',
        action='store',
        dest='min_median_nCount_per_samp_gut',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-samp_nGene_blood', '--min_median_nGene_per_samp_blood',
        action='store',
        dest='min_median_nGene_per_samp_blood',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-samp_nGene_gut', '--min_median_nGene_per_samp_gut',
        action='store',
        dest='min_median_nGene_per_samp_gut',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-max_ncell', '--max_ncells_per_sample',
        action='store',
        dest='max_ncells_per_sample',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-min_ncell', '--min_ncells_per_sample',
        action='store',
        dest='min_ncells_per_sample',
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
        '-fcps', '--filt_cells_pre_samples',
        action='store',
        dest='filt_cells_pre_samples',
        required=True,
        help=''
    )   
    
    parser.add_argument(
        '-ppsqc', '--plot_per_samp_qc',
        action='store',
        dest='plot_per_samp_qc',
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
        '-hvgw', '--hvgs_within',
        action='store',
        dest='hvgs_within',
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
    
    parser.add_argument(
        '-psrt', '--per_samp_relative_threshold',
        action='store',
        dest='per_samp_relative_threshold',
        required=False,
        help=''
    )
    
    parser.add_argument(
        '-slg', '--sample_level_grouping',
        action='store',
        dest='sample_level_grouping',
        required=False,
        help=''
    )
    
    parser.add_argument(
        '-csrf', '--cols_sample_relative_filter',
        action='store',
        dest='cols_sample_relative_filter',
        required=False,
        help=''
    )
    
    parser.add_argument(
        '-pm', '--pref_matrix',
        action='store',
        dest='pref_matrix',
        required=True,
        help=''
    )

    parser.add_argument(
        '-cht', '--calc_hvgs_together',
        action='store',
        dest='calc_hvgs_together',
        required=False,
        help=''
    )

    return parser.parse_args()


def main():
    # Parse options
    inherited_options = parse_options()
    input_file = inherited_options.input_file
    print(input_file)
    discard_other_inflams = inherited_options.discard_other_inflams
    all_blood_immune = inherited_options.all_blood_immune
    min_nUMI = float(inherited_options.min_nUMI)
    nMad_directionality = inherited_options.nMad_directionality
    threshold_method = inherited_options.threshold_method
    relative_grouping = inherited_options.relative_grouping
    relative_nMAD_threshold = float(inherited_options.relative_nMAD_threshold)
    min_nGene = float(inherited_options.min_nGene)
    MTgut = float(inherited_options.MT_thresh_gut)
    MTblood = float(inherited_options.MT_thresh_blood)
    use_MT_thresh_blood_gut_immune = inherited_options.use_MT_thresh_blood_gut_immune
    min_median_nCount_per_samp_blood = float(inherited_options.min_median_nCount_per_samp_blood)
    min_median_nCount_per_samp_gut = float(inherited_options.min_median_nCount_per_samp_gut)
    min_median_nGene_per_samp_blood = float(inherited_options.min_median_nGene_per_samp_blood)
    min_median_nGene_per_samp_gut = float(inherited_options.min_median_nGene_per_samp_gut)
    max_ncells_per_sample = float(inherited_options.max_ncells_per_sample)
    min_ncells_per_sample = float(inherited_options.min_ncells_per_sample)
    use_abs_per_samp = inherited_options.use_abs_per_samp
    filt_cells_pre_samples = inherited_options.filt_cells_pre_samples
    plot_per_samp_qc = inherited_options.plot_per_samp_qc
    filt_blood_keras = inherited_options.filt_blood_keras
    n_variable_genes = float(inherited_options.n_variable_genes)
    hvgs_within = inherited_options.hvgs_within
    remove_problem_genes = inherited_options.remove_problem_genes
    per_samp_relative_threshold = inherited_options.per_samp_relative_threshold
    sample_level_grouping = inherited_options.sample_level_grouping
    cols_sample_relative_filter = inherited_options.cols_sample_relative_filter
    cols_sample_relative_filter = cols_sample_relative_filter.split(",")
    pref_matrix = inherited_options.pref_matrix
    calc_hvgs_together = inherited_options.calc_hvgs_together
    
    print("~~~~~~~~~ Running arguments ~~~~~~~~~")
    print(f"input_file={input_file}")
    print(f"discard_other_inflams={discard_other_inflams}")
    print(f"all_blood_immune={all_blood_immune}")
    print(f"min_nUMI={min_nUMI}")
    print(f"relative_grouping={relative_grouping}")
    print(f"relative_nMAD_threshold={relative_nMAD_threshold}")
    print(f"nMad_directionality: {nMad_directionality}")
    print(f"threshold_method: {threshold_method}")
    print(f"min_nGene={min_nGene}")
    print(f"MTgut={MTgut}")
    print(f"MTblood={MTblood}")
    print(f"use_MT_thresh_blood_gut_immune: {use_MT_thresh_blood_gut_immune}")
    print(f"min_median_nCount_per_samp_blood={min_median_nCount_per_samp_blood}")
    print(f"min_median_nCount_per_samp_gut={min_median_nCount_per_samp_gut}")
    print(f"min_median_nGene_per_samp_blood={min_median_nGene_per_samp_blood}")
    print(f"min_median_nGene_per_samp_gut={min_median_nGene_per_samp_gut}")
    print(f"max_ncells_per_sample: {max_ncells_per_sample}")
    print(f"min_ncells_per_sample:{min_ncells_per_sample}")
    print(f"use_abs_per_samp={use_abs_per_samp}")
    print(f"filt_cells_pre_samples:{filt_cells_pre_samples}")
    print(f"filt_blood_keras={filt_blood_keras}")
    print(f"n_variable_genes={n_variable_genes}")
    print(f"hvgs_within: {hvgs_within}")
    print(f"remove_problem_genes={remove_problem_genes}")
    print(f"per_samp_relative_threshold:{per_samp_relative_threshold}")
    print(f"sample_level_grouping:{sample_level_grouping}")
    print(f"cols_sample_relative_filter:{cols_sample_relative_filter}")
    print(f"pref_matrix: {pref_matrix}")
    print("Parsed args")
    
    # Finally, derive and print the tissue arguments
    tissue=inherited_options.tissue
    print(f"~~~~~~~ TISSUE:{tissue}")
    
    # Do we have a GPU?
    use_gpu = torch.cuda.is_available()
    print(f"Is there a GPU available?: {use_gpu}")

    # Get basedir - modified to run in current dir
    # outdir = os.path.dirname(os.path.commonprefix([input_file]))
    # outdir = os.path.dirname(os.path.commonprefix([outdir]))
    outdir = "results"
    if os.path.exists(outdir) == False:
        os.mkdir(outdir)

    # Update the outdir path to include the tissue
    outdir = f"{outdir}/{tissue}"
    if os.path.exists(outdir) == False:
        os.mkdir(outdir)

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

    print("Set up outdirs")

    # Define global figure directory
    sc.settings.figdir=figpath
    sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")

    # Load the given files and perform initial formatting / filters
    adata = sc.read_h5ad(input_file)
    # make sure we have counts as .X
    #if "counts" in adata.layers.keys():
    #    adata.X = adata.layers['counts'].copy() # Takes lots of memory, .X is already counts

    print(f"Initial shape of the data is:{adata.shape}")
    tissues = np.unique(adata.obs['tissue'])
    #for t in tissues:
    #    tempt = adata.obs[adata.obs['tissue'] == t]
    #    print(f"For {t}, there is {len(np.unique(tempt['sanger_sample_id']))} samples. A total of {tempt.shape[0]} cells")
    #    cd = tempt[tempt['disease_status'] == "CD"]
    #    print(f"{len(np.unique(cd['sanger_sample_id']))} are CD")
    #    del tempt
        
    # Also save the gene var df
    #adata.var.to_csv(f"results/{tissue}/tables/raw_gene.var.txt", sep = "\t")

    # Discard other inflams if wanted to 
    if discard_other_inflams == "yes":
        adata = adata[adata.obs['disease_status'].isin(['healthy', 'cd'])]
        print(f"Shape of adata after CD/healthy subset = {adata.shape}")
        
    ####################################
    ######### Preliminary QC ###########
    ####################################
    print("~~~~~~~~~~~~ Conducting preliminary QC ~~~~~~~~~~~~~")
    qc_path = f"{figpath}/QC"
    if os.path.exists(qc_path) == False:
        os.mkdir(qc_path)
        
    # Gather the input options so they can be passed to the wrapper functions
    cols = ["pct_counts_gene_group__mito_transcript", "log_n_genes_by_counts", "log_total_counts"]
    thresholds = {"pct_counts_gene_group__mito_transcript": {"blood": MTblood, "r": MTgut, "ti": MTgut}, "log_n_genes_by_counts": np.log10(min_nGene), "log_total_counts": np.log10(min_nUMI)}
    over_under = {"pct_counts_gene_group__mito_transcript": "under", "log_n_genes_by_counts": "over", "log_total_counts": "over"}
    
    # Calculate log of QC metrics:
    adata.obs['log_n_genes_by_counts'] = np.log10(adata.obs['n_genes_by_counts'])
    adata.obs['log_total_counts'] = np.log10(adata.obs['total_counts'])
    
    # Remove samples with fewer than X cells. Important if doing relative QC
    adata.obs['samp_tissue'] = adata.obs['sanger_sample_id'].astype('str') + "_" + adata.obs['tissue'].astype('str')
    samp_data = np.unique(adata.obs.samp_tissue, return_counts=True)
    cells_sample = pd.DataFrame({'sample': samp_data[0], 'Ncells':samp_data[1]})
    chuck = cells_sample.loc[cells_sample['Ncells'] < min_ncells_per_sample, 'sample'].values
    if len(chuck) > 0:
        adata = adata[~adata.obs['samp_tissue'].isin(chuck)]
        
    # If running on epithelial, remove samples from the blood
    if tissue == "all_Epithelial":
        adata = adata[adata.obs['tissue'] != "blood"]
    
    # Plot prelimeinary exploratory plots
    # These are per-grouping for each parameter - to highlight where shared/common QC could be useful
    # Also force this to be done per tissue
    for index, c in enumerate(cols):
        nmad_qc.dist_plot(adata, c, within="tissue", relative_threshold=relative_nMAD_threshold, absolute=thresholds[c], out=qc_path)
        nmad_qc.dist_plot(adata, c, within=relative_grouping, relative_threshold=relative_nMAD_threshold, absolute=thresholds[c], out=qc_path)
    
    # Now calculate the QC metrics, apply absolute/relative thresholds with the desired method, and save plots of where the thresholds lie based on this grouping
    for index, c in enumerate(cols):
        nmad_qc.update_obs_qc_plot_thresh(adata, c, within=relative_grouping, relative_threshold=relative_nMAD_threshold, threshold_method = threshold_method, relative_directionality = "bi", absolute=thresholds[c], absolute_directionality = over_under[c], plot =True, out=qc_path, out_suffix = "_THRESHOLDED")

    # Extract list of high and low cell number samples - PRE QC
    high_samps = np.unique(cells_sample.loc[cells_sample.Ncells > max_ncells_per_sample, "sample"])
    low_samps = np.array(cells_sample.loc[cells_sample.Ncells < min_ncells_per_sample, "sample"])

    # If filtering sequentially, filter cells at this point BEFORE the filtering of samples
    if filt_cells_pre_samples == "yes": 
        keep_columns = adata.obs.filter(like='_keep').columns
        adata = adata[adata.obs[keep_columns].all(axis=1)]
    
    ##############################################
    #### Now address the sample-level QC #########
    ##############################################
    
    samp_data = np.unique(adata.obs.samp_tissue, return_counts=True)
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    sns.distplot(cells_sample.Ncells)
    plt.xlabel('Cells/sample')
    plt.axvline(x = 500, color = 'red', linestyle = '--', alpha = 0.5)
    ax.set(xlim=(0, max(cells_sample.Ncells)))
    plt.savefig(f"{qc_path}/sample_cells_per_sample_all.png", bbox_inches='tight')
    plt.clf()

    # Also do this within tissue
    for t in tissues:
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
        sns.distplot(cells_sample[cells_sample['sample'].isin(adata.obs[adata.obs['tissue'] == t]['samp_tissue'])].Ncells)
        plt.xlabel('Cells/sample')
        plt.axvline(x = 500, color = 'red', linestyle = '--', alpha = 0.5)
        ax.set(xlim=(0, max(cells_sample.Ncells)))
        plt.savefig(f"{qc_path}/sample_cells_per_sample_{t}.png", bbox_inches='tight')
        plt.clf()

    # Summarise
    depth_count = pd.DataFrame(index = np.unique(adata.obs.samp_tissue), columns=["Mean_nCounts", "nCells", "High_cell_sample", "n_genes_by_counts", "Median_nCounts", "Median_nGene_by_counts", "Median_MT"])
    for s in range(0, depth_count.shape[0]):
        samp = depth_count.index[s]
        depth_count.iloc[s,1] = adata.obs[adata.obs.samp_tissue == samp].shape[0]
        depth_count.iloc[s,0] = sum(adata.obs[adata.obs.samp_tissue == samp].total_counts)/depth_count.iloc[s,1]
        depth_count.iloc[s,3] = sum(adata.obs[adata.obs.samp_tissue == samp].n_genes_by_counts)/depth_count.iloc[s,1]
        depth_count.iloc[s,4] = np.median(adata.obs[adata.obs.samp_tissue == samp].total_counts)
        depth_count.iloc[s,5] = np.median(adata.obs[adata.obs.samp_tissue == samp].n_genes_by_counts)
        depth_count.iloc[s,6] = np.median(adata.obs[adata.obs.samp_tissue == samp].pct_counts_gene_group__mito_transcript)
        if samp in high_samps:
            depth_count.iloc[s,2] = "Red"
        else: 
            depth_count.iloc[s,2] = "Navy"
        # Also annotate the samples with low number of cells - Are these sequenced very deeply?
        if samp in low_samps:
            depth_count.iloc[s,2] = "Green"

    depth_count["log10_Mean_Counts"] = np.log10(np.array(depth_count["Mean_nCounts"].values, dtype = "float"))
    depth_count["log10_Median_nCounts"] = np.log10(np.array(depth_count["Median_nCounts"].values, dtype = "float"))
    depth_count.to_csv(f"{tabpath}/depth_count_pre_cell_filtration.csv")

    # Plot the distribution per tissue
    cols_to_plot = ["nCells", "Median_nCounts", "Median_nGene_by_counts", "Median_MT"]
    adata.obs['disease_tissue'] = adata.obs['tissue'].astype(str) + "_" + adata.obs['disease_status'].astype(str)
    td = np.unique(adata.obs['disease_tissue'])
    if plot_per_samp_qc == "yes":
        for c in cols_to_plot:
            print(c)
            plt.figure(figsize=(8, 6))
            for t in tissues:
                samps = np.unique(adata.obs.loc[adata.obs['tissue'] == t,"samp_tissue"].values)
                data = depth_count[depth_count.index.isin(samps)][c]
                sns.distplot(data, hist=False, rug=True, label=t)
            
            plt.legend()
            plt.xlabel(f'{c}/sample')
            plt.savefig(f"{qc_path}/sample_dist_{c}.png", bbox_inches='tight')
            plt.clf()
            
        # Also have a look by tissue x disease
        td = np.unique(adata.obs['disease_tissue'])
        for c in cols_to_plot:
            print(c)
            plt.figure(figsize=(8, 6))
            for t in td:
                samps = np.unique(adata.obs.loc[adata.obs['disease_tissue'] == t, 'samp_tissue'].values)
                data = depth_count[depth_count.index.isin(samps)][c]
                sns.distplot(data, hist=False, rug=True, label=t)
            #
            plt.legend()
            plt.xlabel(f'{c}/sample')
            plt.savefig(f"{qc_path}/sample_dist_{c}_tissue_disease.png", bbox_inches='tight')
            plt.clf()
        
        # Within tissue for median
        for t in tissues:
            samps = np.unique(adata.obs.loc[adata.obs['tissue'] == t, 'samp_tissue'].values)
            plt.figure(figsize=(8, 6))
            plt.scatter(depth_count[depth_count.index.isin(samps)]["Median_nCounts"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["High_cell_sample"], alpha=0.7)
            plt.xlabel('Median counts / cell')
            plt.ylabel('Median genes detected / cell')
            if t == "blood":
                plt.axvline(x = min_median_nCount_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axhline(y = min_median_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
                plt.title(f"{t} - min_median_nCount: {min_median_nCount_per_samp_blood}, min_median_nGene: {min_median_nGene_per_samp_blood}")
            else:
                plt.axvline(x = min_median_nCount_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
                plt.title(f"{t} - min_median_nCount: {min_median_nCount_per_samp_gut}, min_median_nGene: {min_median_nGene_per_samp_gut}")
            
            plt.savefig(f"{qc_path}/sample_median_counts_ngenes_{t}.png", bbox_inches='tight')
            plt.clf()
            
        # Within disease x tissue
        for t in td:
            samps = np.unique(adata.obs.loc[adata.obs['disease_tissue'] == t, 'samp_tissue'].values)
            plt.figure(figsize=(8, 6))
            plt.scatter(depth_count[depth_count.index.isin(samps)]["Median_nCounts"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["High_cell_sample"], alpha=0.7)
            plt.xlabel('Median counts / cell')
            plt.ylabel('Median genes detected / cell')
            if t == "blood":
                plt.axvline(x = min_median_nCount_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axhline(y = min_median_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
                plt.title(f"{t} - min_median_nCount: {min_median_nCount_per_samp_blood}, min_median_nGene: {min_median_nGene_per_samp_blood}")
            else:
                plt.axvline(x = min_median_nCount_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
                plt.title(f"{t} - min_median_nCount: {min_median_nCount_per_samp_gut}, min_median_nGene: {min_median_nGene_per_samp_gut}")
            
            plt.savefig(f"{qc_path}/sample_median_counts_ngenes_{t}.png", bbox_inches='tight')
            plt.clf()
        
        # median nGenes vs nCells
        for t in td:
            samps = np.unique(adata.obs.loc[adata.obs['disease_tissue'] == t, 'samp_tissue'].values)
            plt.figure(figsize=(8, 6))
            plt.scatter(depth_count[depth_count.index.isin(samps)]["nCells"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["High_cell_sample"], alpha=0.7)
            plt.xlabel('nCells')
            plt.ylabel('Median genes detected / cell')
            if t == "blood":
                plt.axvline(x = min_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axvline(x = max_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axhline(y = min_median_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axvline(x = np.median(depth_count[depth_count.index.isin(samps)]["nCells"]), color = 'black', linestyle = '--', alpha = 0.5)
                plt.axhline(y = np.median(depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"]), color = 'black', linestyle = '--', alpha = 0.5)
                plt.title(f"{t} - {min_ncells_per_sample} < nCells < {max_ncells_per_sample}, median_nGene > {min_median_nGene_per_samp_blood}. Black = medians")
            else:
                plt.axvline(x = min_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axvline(x = max_ncells_per_sample, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
                plt.axvline(x = np.median(depth_count[depth_count.index.isin(samps)]["nCells"]), color = 'black', linestyle = '--', alpha = 0.5)
                plt.axhline(y = np.median(depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"]), color = 'black', linestyle = '--', alpha = 0.5)
                plt.title(f"{t} - {min_ncells_per_sample} < nCells < {max_ncells_per_sample}, median_nGene >  {min_median_nGene_per_samp_gut}. Black = medians")
            
            plt.savefig(f"{qc_path}/sample_ncells_median_ngenes_{t}.png", bbox_inches='tight')
            plt.clf()

    # These are clearly different across tissues
    # Either apply absolute thresholds or a relative cut off, or both
    # If per-samp relative threshold, compute and filter
    if per_samp_relative_threshold == "yes":
        print("Filtering on the basis of relative sample thresholds")
        # NOTE: This will overwrite the per-sample absolute cut off defined above
        # Calculate log10
        cols_to_log = ['Median_nCounts', 'Median_nGene_by_counts', 'Median_MT']
        for c in cols_to_log:
            depth_count[f"log10_{c}"] = np.log10(np.array(depth_count[c].values, dtype = "float"))
        
        # Calculate nMad and what would be kept upon thresholding of the desired columns
        for c in cols_sample_relative_filter:
            print(c)
            if sample_level_grouping is not None:
                if sample_level_grouping not in depth_count.columns:
                    # Make sure this is on the depth_count dataframe
                    to_add = adata.obs.reset_index()[["samp_tissue", sample_level_grouping]].drop_duplicates()
                    depth_count.reset_index(inplace=True)
                    depth_count.rename(columns={"index": "samp_tissue"}, inplace=True)
                    depth_count = depth_count.merge(to_add, how="left", on="samp_tissue")
                    depth_count.set_index("samp_tissue", inplace=True)
                #
                depth_count[f"nMad_{c}"] = nmad_append(depth_count, c, sample_level_grouping)
            else:
                depth_count[f"nMad_{c}"] = nmad_append(depth_count, c)
            
            if nMad_directionality == "uni":
                if "MT" in c:
                    depth_count[f'keep_nMad_{c}'] = depth_count[f"nMad_{c}"] < relative_nMAD_threshold
                else:
                    depth_count[f'keep_nMad_{c}'] = depth_count[f"nMad_{c}"] > relative_nMAD_threshold
            else:
                depth_count[f'keep_nMad_{c}'] = np.abs(depth_count[f"nMad_{c}"]) < relative_nMAD_threshold
            
            depth_count[f'keep_nMad_{c}'] = depth_count[f'keep_nMad_{c}'].astype(str)
        
        # Plot
        for c in cols_sample_relative_filter:
            if sample_level_grouping is not None:
                groups = np.unique(adata.obs[sample_level_grouping])
                for t in groups:
                    print(f"{t} - {c}")
                    data = depth_count[depth_count[sample_level_grouping] == t]
                    data = data[c]
                    absolute_diff = np.abs(data - np.median(data))
                    mad = np.median(absolute_diff)
                    cutoff_low = np.median(data) - (float(relative_nMAD_threshold) * mad)
                    cutoff_high = np.median(data) + (float(relative_nMAD_threshold) * mad)
                    if "log" in c:
                        plot = sns.distplot(data, hist=False, rug=True, label=f'{t} - (relative): {10**cutoff_low:.2f}-{10**cutoff_high:.2f}')
                    else:
                        plot = sns.distplot(data, hist=False, rug=True, label=f'{t} - (relative): {cutoff_low:.2f}-{cutoff_high:.2f}')
                    #
                    line_color = plot.get_lines()[-1].get_color()
                    plt.axvline(x = cutoff_low, linestyle = '--', alpha = 0.5, color=line_color)
                    plt.axvline(x = cutoff_high, linestyle = '--', alpha = 0.5, color=line_color)
                #
            else:
                print(c)
                data = data[c]
                absolute_diff = np.abs(data - np.median(data))
                mad = np.median(absolute_diff)
                cutoff_low = np.median(data) - (float(relative_nMAD_threshold) * mad)
                cutoff_high = np.median(data) + (float(relative_nMAD_threshold) * mad)
                if "log" in c:
                    sns.distplot(data, hist=False, rug=True, label=f'(relative): {10**cutoff_low:.2f}-{10**cutoff_high:.2f}')
                else:
                    sns.distplot(data, hist=False, rug=True, label=f'(relative): {cutoff_low:.2f}-{cutoff_high:.2f}')
                #
                plt.axvline(x = cutoff_low, linestyle = '--', alpha = 0.5)
                plt.axvline(x = cutoff_high, linestyle = '--', alpha = 0.5)
            #
            plt.xlabel(f"Sample-level: {c}")
            plt.legend()
            total_cells = sum(depth_count['nCells'])
            total_samps = depth_count.shape[0]
            subset = depth_count[depth_count[f'keep_nMad_{c}'] == "True"]
            nkeep_cells = sum(subset['nCells'])
            nkeep_samples = subset.shape[0]
            plt.title(f"Loss of {100*((total_cells - nkeep_cells)/total_cells):.2f}% cells, {100*((total_samps - nkeep_samples)/total_samps):.2f}% samples")
            plt.savefig(f"{qc_path}/sample_dist_{c}_distribution_thresholded.png", bbox_inches='tight')
            plt.clf()
            
        # Apply the thresholds
        filt_column_names = ['keep_nMad_' + col for col in cols_sample_relative_filter]
        for c in filt_column_names:
            print(f"Booleanising {c}")
            depth_count[c] = depth_count[c] == "True" # Make sure they are boolean
        
        boolean_series = depth_count[filt_column_names].all(axis=1)
        depth_count['samples_keep'] = boolean_series
            
    # Have a look at samples that would be lost vs the rest based on the absolute min median nGene filter
    if "disease_tissue" not in depth_count.columns:
        # Make sure this is on the depth_count dataframe
        to_add = adata.obs.reset_index()[["samp_tissue", "disease_tissue"]].drop_duplicates()
        depth_count.reset_index(inplace=True)
        depth_count.rename(columns={"index": "samp_tissue"}, inplace=True)
        depth_count = depth_count.merge(to_add, how="left", on="samp_tissue")
        depth_count.set_index("samp_tissue", inplace=True)
    
    for t in td: 
        print(t)
        temp = depth_count[depth_count['disease_tissue'] == t]
        # Make stacked bar of the keras annotations
        if "blood" in t:
            exclude = temp[temp['Median_nGene_by_counts'] < min_median_nGene_per_samp_blood].index.values
        else:
            exclude = temp[temp['Median_nGene_by_counts'] < min_median_nGene_per_samp_gut].index.values
        
        if plot_per_samp_qc == "yes":
            proportion_df = adata.obs.groupby(['samp_tissue', 'category__machine']).size().reset_index(name='count')
            proportion_df = proportion_df[proportion_df['samp_tissue'].isin(temp.index.values)]
            proportion_df['proportion'] = proportion_df.groupby('samp_tissue')['count'].transform(lambda x: x / x.sum())
            pivot_df = proportion_df.pivot(index='samp_tissue', columns='category__machine', values='proportion').fillna(0)
            colors = sns.color_palette("husl", len(pivot_df.columns))
            fig, ax = plt.subplots(figsize=(10, 6))
            excluded_df = pivot_df.loc[exclude]
            remaining_df = pivot_df.drop(exclude)
            if remaining_df.shape[0] > 50:
                remaining_df = remaining_df.sample(n=50, random_state=17)
        
            combined_df = pd.concat([excluded_df, pd.DataFrame([np.nan] * len(pivot_df.columns)).T, remaining_df])
            combined_df.index = list(exclude) + [''] + list(remaining_df.index)
            bottom = np.zeros(len(combined_df))
            for idx, category in enumerate(pivot_df.columns):
                ax.bar(combined_df.index, combined_df[category], bottom=bottom, color=colors[idx], label=category)
                bottom += combined_df[category].fillna(0).values
            #
            ax.legend(title='Category__Machine', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_title('Relative Proportions of Category__Machine by Samp_Tissue')
            ax.set_xlabel('Samp_Tissue')
            ax.set_ylabel('Proportion')
            ax.set_xticks([])
            plt.title(f"{t} : Left = excluded, Right = 50 non-excluded samples")
            plt.savefig(f"{qc_path}/category_proportions_across_{t}_samples_absolute_min_nGene.png", bbox_inches='tight')
            plt.clf()
            # Also make a PCA and colour by whether the samples are in the excluded set or not
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(pivot_df)
            pca_df = pd.DataFrame(data=pca_result[:,:2], columns=['PC1', 'PC2'], index=pivot_df.index)
            plt.figure(figsize=(10, 8))
            plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
            # Save
            pca_df.to_csv(f"{tabpath}/pca_cell_contributions_{t}.csv")
            #
            # Highlight bad samples
            for sample in exclude:
                if sample in pca_df.index:
                    plt.scatter(pca_df.loc[sample, 'PC1'], pca_df.loc[sample, 'PC2'], color='red', label='Maybe bad Sample')
                    plt.annotate(sample, (pca_df.loc[sample, 'PC1'], pca_df.loc[sample, 'PC2']))
            #
            plt.xlabel('Cell contribution PC 1')
            plt.ylabel('Cell contribution PC 1')
            plt.title(f'{t} - PCA of category__machine proportions')
            plt.savefig(f"{qc_path}/pca_cell_contributions_{t}_absolute_min_nGene.png", bbox_inches='tight')
            plt.clf()        
        
    if use_abs_per_samp == "yes":
        # Min median nGene
        blood_keep = (depth_count['Median_nCounts'] > min_median_nCount_per_samp_blood) & (depth_count['Median_nGene_by_counts'] > min_median_nGene_per_samp_blood)
        blood_keep = blood_keep & depth_count.index.isin(adata.obs[adata.obs['tissue'] == "blood"]['samp_tissue'])
        blood_keep = blood_keep[blood_keep == True].index
        gut_keep = (depth_count['Median_nCounts'] > min_median_nCount_per_samp_gut) & (depth_count['Median_nGene_by_counts'] > min_median_nGene_per_samp_gut)
        gut_keep = gut_keep & depth_count.index.isin(adata.obs[adata.obs['tissue'] != "blood"]['samp_tissue'])
        gut_keep = gut_keep[gut_keep == True].index
        both_keep = list(np.concatenate([blood_keep, gut_keep]))
        if "samples_keep" not in depth_count.columns:
            depth_count['samples_keep'] = depth_count.index.isin(both_keep)
        else:
            blood_condition = ['blood' in s for s in depth_count['disease_tissue']]
            gut_condition = ['blood' not in s for s in depth_count['disease_tissue']]
            depth_count.loc[blood_condition & (depth_count['Median_nGene_by_counts'] < min_median_nGene_per_samp_blood), 'samples_keep'] = False
            depth_count.loc[gut_condition & (depth_count['Median_nGene_by_counts'] < min_median_nGene_per_samp_gut), 'samples_keep'] = False
        #
        # nCells
        depth_count.loc[(depth_count['nCells'] < min_ncells_per_sample) | (depth_count['nCells'] > max_ncells_per_sample), 'samples_keep'] = False
        lost = depth_count.shape[0]-(sum(depth_count['samples_keep']))
        print(f"The number of samples beng lost is {lost}")  
        
    # Alter cols
    depth_count.reset_index(inplace=True)
    depth_count.rename(columns={"index": "samp_tissue", "n_genes_by_counts": "mean_n_genes_by_counts"}, inplace=True)
    # Save
    depth_count.to_csv(f"{tabpath}/depth_count_pre_cell_filtration2.csv", index=False)
    
    # bind to anndata
    adata.obs.reset_index(inplace=True)
    adata.obs = adata.obs.merge(depth_count.drop(columns="disease_tissue"), how="left", on="samp_tissue")
    columns_to_convert = [col for col in adata.obs.columns if col in depth_count.columns and col != 'High_cell_sample' and col != "samp_tissue" and col != "samples_keep" and col != "disease_tissue"]
    print(f"converting: {columns_to_convert}")
    adata.obs[columns_to_convert] = adata.obs[columns_to_convert].astype(float)
    adata.obs.set_index("cell", inplace=True)
    
    # Plot sankey
    #temp = adata.obs[["log_total_counts_keep", "log_n_genes_by_counts_keep", "pct_counts_gene_group__mito_transcript_keep", "samples_keep"]]
    #for col in temp.columns:
    #    temp[col] = col + "-" + temp[col].astype(str)
    #
    #temp['input'] = "input"
    #grouped = temp.groupby(["input", "log_total_counts_keep"])
    #input_to_total_counts = grouped.size()#.unstack().div(grouped.size().unstack().sum(axis=1), axis=0).stack()
    #grouped = temp.groupby(["log_total_counts_keep", "log_n_genes_by_counts_keep"])
    #total_counts_to_n_genes = grouped.size()#.unstack().div(grouped.size().unstack().sum(axis=1), axis=0).stack()
    ## Group by unique combinations of "log_n_genes_by_counts_keep" and "pct_counts_gene_group__mito_transcript_keep"
    #grouped = temp.groupby(["log_n_genes_by_counts_keep", "pct_counts_gene_group__mito_transcript_keep"])
    ## Calculate the percentage of each value of "log_n_genes_by_counts_keep" that contributes to each value of "pct_counts_gene_group__mito_transcript_keep"
    #n_genes_to_MT_perc = grouped.size()#.unstack().div(grouped.size().unstack().sum(axis=1), axis=0).stack()
    ## Group by unique combinations of "pct_counts_gene_group__mito_transcript_keep" and "samples_keep"
    #grouped = temp.groupby(["pct_counts_gene_group__mito_transcript_keep", "samples_keep"])
    ## Calculate the percentage of each value of "pct_counts_gene_group__mito_transcript_keep" that contributes to each value of "samples_keep"
    #MT_perc_to_samples = grouped.size()#.unstack().div(grouped.size().unstack().sum(axis=1), axis=0).stack()
    ## Concatenate the three DataFrames vertically
    #result = pd.concat([input_to_total_counts, total_counts_to_n_genes, n_genes_to_MT_perc, MT_perc_to_samples]).reset_index()
    #result.columns = ['source', 'target', 'value']                               
    #
    #print("result")
    #print(result)
    #sankey = hv.Sankey(result, label='Retention of cells on the basis of different metrics')
    #sankey.opts(label_position='left', edge_color='target', node_color='index', cmap='tab20')
    #hv.output(fig='png')
    #hv.save(sankey, f"{qc_path}/sankey_cell_retention_across_QC.png")
    
    # Apply sample-level threshold
    adata = adata[adata.obs['samples_keep'] == True ]
    
    ####################################
    #### Expression Normalisation ######
    ####################################
    print("~~~~~~~~~~~~ Conducting expression normalisation ~~~~~~~~~~~~~")
    sc.pp.filter_genes(adata, min_cells=5)
    print("Filtered genes")

    # Keep a copy of the raw counts
    if "counts" not in adata.layers.keys():
        adata.layers['counts'] = adata.X.copy()
        print("Copied counts")

    if "log1p_cp10k" not in adata.layers.keys():
        # Calulate the CP10K expression
        sc.pp.normalize_total(adata, target_sum=1e4)
        print("CP10K")
        # Now normalise the data to identify highly variable genes (Same as in Tobi and Monika's paper)
        sc.pp.log1p(adata)
        print("log1p")
        # Store this (for later, e.g 005 and plotting)
        adata.layers['log1p_cp10k'] = adata.X.copy()
    else:
        print("log1p_cp10k already calculated")
    
    # Save this 
    tissue=inherited_options.tissue
    #sparse_matrix = sp.sparse.csc_matrix(adata.layers['log1p_cp10k'])
    #sp.sparse.save_npz(f"results/{tissue}/tables/log1p_cp10k_sparse.npz", sparse_matrix)
    #print("Saved sparse matrix")
    # Save index and columns
    #with open(f"results/{tissue}/tables/cells.txt", 'w') as file:
    #    for index, row in adata.obs.iterrows():
    #        file.write(str(index) + '\n')
    #
    #print("Saved cells")
    #with open(f"results/{tissue}/tables/genes.txt", 'w') as file:
    #    for index, row in adata.var.iterrows():
    #        file.write(str(index) + '\n')
    #
    #print("Saved genes")
    #del sparse_matrix
    
    adata.write_h5ad(f"results/{tissue}/objects/adata_unfilt_log1p_cp10k.h5ad")
    print("Saved after log1p")
    
    # identify highly variable genes and scale these ahead of PCA
    print("Calculating highly variable")
    if hvgs_within == "None":
        sc.pp.highly_variable_genes(adata, flavor="seurat", layer="log1p_cp10k", n_top_genes=int(n_variable_genes), subset=True)
    else:
        sc.pp.highly_variable_genes(adata, flavor="seurat", layer="log1p_cp10k", batch_key=hvgs_within, n_top_genes=int(n_variable_genes), subset=True)
    print("Found highly variable")

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

    # Print the final shape of the high QC data
    print(f"The final shape of the high QC data is: {adata.shape}")
    for t in tissues:
     tempt = adata.obs[adata.obs['tissue'] == t]
     print(f"For {t}, there is {len(np.unique(tempt['sanger_sample_id']))} samples. A total of {tempt.shape[0]} cells")
     cd = tempt[tempt['disease_status'] == "CD"]
     print(f"{len(np.unique(cd['sanger_sample_id']))} are CD")

    if pref_matrix not in ["scVI", "scANVI"]:
        # Only need to compute PCA if we know the data is going to be Harmony'd, or bbknn'd
        ####################################
        ######### PCA calculation ##########
        ####################################

        # Scale
        sc.pp.scale(adata, max_value=10)
        print("Scaled")

        # Run PCA
        sc.tl.pca(adata, svd_solver='arpack')

        # Plot PCA
        sc.pl.pca(adata, color="tissue", save="_tissue.png")
        #sc.pl.pca(adata, color="category__machine", save="_category.png")
        #sc.pl.pca(adata, color="input", save="_input.png")
        sc.pl.pca(adata, color=relative_grouping, save=f"_{relative_grouping}.png")

        # PLot Elbow plot
        sc.pl.pca_variance_ratio(adata, log=True, save=True, n_pcs = 50)

        #  Determine the optimimum number of PCs
        # Extract PCs
        pca_variance=pd.DataFrame({'x':list(range(1, 51, 1)), 'y':adata.uns['pca']['variance']})
        # Identify 'knee'
        knee=kd.KneeLocator(x=list(range(1, 51, 1)), y=adata.uns['pca']['variance_ratio'], curve="convex", direction = "decreasing")
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
        loadings.to_csv(tabpath + "/PCA_loadings.csv")

        # Save the PCA matrix 
        pca = pd.DataFrame(adata.obsm['X_pca'])
        pca = pca.iloc[:,0:knee_point]
        pca.index = adata.obs.index
        pca.to_csv(tabpath + "/PCA_up_to_knee.csv")
    else:
        # Save dummy knee
        from numpy import asarray
        from numpy import savetxt
        savetxt(tabpath + "/knee.txt", asarray([[0]]), delimiter='\t')

    # Save object ahead of batch correction
    adata.write(objpath + "/adata_PCAd.h5ad")


# Execute
if __name__ == '__main__':
    main()
