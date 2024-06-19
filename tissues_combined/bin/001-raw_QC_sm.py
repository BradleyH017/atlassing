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
hv.extension('matplotlib')
print("Loaded libraries")

########### Define custom functions used here ###############
def nmad_calc(data):
    absolute_diff = np.abs(data - np.median(data))
    mad = np.median(absolute_diff)
    # Use direction aware filtering
    nmads = (data - np.median(data))/mad
    return(nmads)

def nmad_append(df, var, group=[]):
    # Calculate
    if group:
        vals = df.groupby(group)[var].apply(nmad_calc)
        # Return to df
        temp = pd.DataFrame(vals)
        temp.reset_index(inplace=True)
        if "level_1" in temp.columns:
            temp.set_index("level_1", inplace=True)
        else: 
            index_name=df.reset_index().columns[0]
            temp.set_index(index_name, inplace=True)
        
        #temp = temp[var]
        temp = temp.reindex(df.index)
        return(temp[var])
    else:
        vals = nmad_calc(df[var])
        return(vals)

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
        '-abs_nUMI', '--use_absolute_nUMI',
        action='store',
        dest='use_absolute_nUMI',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-use_rel_mad', '--use_relative_mad',
        action='store',
        dest='use_relative_mad',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-filter_sequentially', '--filter_sequentially',
        action='store',
        dest='filter_sequentially',
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
        '-pw', '--plot_within',
        action='store',
        dest='plot_within',
        required=True,
        help=''
    )

    parser.add_argument(
        '-lineage_column_name', '--lineage_column',
        action='store',
        dest='lineage_column',
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
        '-rel_nUMI_log', '--relative_nUMI_log',
        action='store',
        dest='relative_nUMI_log',
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
        '-rel_nGene_log', '--relative_nGene_log',
        action='store',
        dest='relative_nGene_log',
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
        '-abs_max_MT', '--absolute_max_MT',
        action='store',
        dest='absolute_max_MT',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-minMT', '--min_MT',
        action='store',
        dest='min_MT',
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

    return parser.parse_args()


def main():
    # Parse options
    inherited_options = parse_options()
    input_file = inherited_options.input_file
    print(input_file)
    discard_other_inflams = inherited_options.discard_other_inflams
    all_blood_immune = inherited_options.all_blood_immune
    min_nUMI = float(inherited_options.min_nUMI)
    use_absolute_nUMI = inherited_options.use_absolute_nUMI
    use_relative_mad = inherited_options.use_relative_mad
    filter_sequentially = inherited_options.filter_sequentially
    nMad_directionality = inherited_options.nMad_directionality
    plot_within = inherited_options.plot_within
    plot_within=plot_within.split(",")
    lineage_column = inherited_options.lineage_column
    relative_grouping = inherited_options.relative_grouping
    relative_grouping = relative_grouping.split(",")
    relative_nMAD_threshold = float(inherited_options.relative_nMAD_threshold)
    relative_nUMI_log = inherited_options.relative_nUMI_log
    min_nGene = float(inherited_options.min_nGene)
    use_absolute_nGene = inherited_options.use_absolute_nGene
    relative_nGene_log = inherited_options.relative_nGene_log
    MTgut = float(inherited_options.MT_thresh_gut)
    MTblood = float(inherited_options.MT_thresh_blood)
    use_absolute_MT = inherited_options.use_absolute_MT
    absolute_max_MT = float(inherited_options.absolute_max_MT)
    min_MT = inherited_options.min_MT
    min_MT = float(min_MT)
    use_MT_thresh_blood_gut_immune = inherited_options.use_MT_thresh_blood_gut_immune
    min_median_nCount_per_samp_blood = float(inherited_options.min_median_nCount_per_samp_blood)
    min_median_nCount_per_samp_gut = float(inherited_options.min_median_nCount_per_samp_gut)
    min_median_nGene_per_samp_blood = float(inherited_options.min_median_nGene_per_samp_blood)
    min_median_nGene_per_samp_gut = float(inherited_options.min_median_nGene_per_samp_gut)
    max_ncells_per_sample = float(inherited_options.max_ncells_per_sample)
    min_ncells_per_sample = float(inherited_options.min_ncells_per_sample)
    use_abs_per_samp = inherited_options.use_abs_per_samp
    filt_cells_pre_samples = inherited_options.filt_cells_pre_samples
    filt_blood_keras = inherited_options.filt_blood_keras
    n_variable_genes = float(inherited_options.n_variable_genes)
    remove_problem_genes = inherited_options.remove_problem_genes
    per_samp_relative_threshold = inherited_options.per_samp_relative_threshold
    sample_level_grouping = inherited_options.sample_level_grouping
    cols_sample_relative_filter = inherited_options.cols_sample_relative_filter
    cols_sample_relative_filter = cols_sample_relative_filter.split(",")
    
    print("~~~~~~~~~ Running arguments ~~~~~~~~~")
    print(f"input_file={input_file}")
    print(f"discard_other_inflams={discard_other_inflams}")
    print(f"all_blood_immune={all_blood_immune}")
    print(f"min_nUMI={min_nUMI}")
    print(f"use_absolute_nUMI={use_absolute_nUMI}")
    print(f"relative_grouping={relative_grouping}")
    print(f"relative_nMAD_threshold={relative_nMAD_threshold}")
    print(f"filter_sequentially: {filter_sequentially}")
    print(f"nMad_directionality: {nMad_directionality}")
    print(f"plot_within: {plot_within}")
    print(f"relative_nUMI_log={relative_nUMI_log}")
    print(f"min_nGene={min_nGene}")
    print(f"use_absolute_nGene={use_absolute_nGene}")
    print(f"relative_nGene_log={relative_nGene_log}")
    print(f"MTgut={MTgut}")
    print(f"MTblood={MTblood}")
    print(f"use_absolute_MT={use_absolute_MT}")
    print(f"absolute_max_MT={absolute_max_MT}")
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
    print(f"remove_problem_genes={remove_problem_genes}")
    print(f"per_samp_relative_threshold:{per_samp_relative_threshold}")
    print(f"sample_level_grouping:{sample_level_grouping}")
    print(f"cols_sample_relative_filter:{cols_sample_relative_filter}")
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
    # make sure we have counts as .X and no others
    if "counts" in adata.layers.keys():
        adata.X = adata.layers['counts'].copy()

    if "lognorm" in adata.layers.keys():
        del adata.layers["lognorm"]

    print(f"Initial shape of the data is:{adata.shape}")

    # Save the raw matrix and gene var before doing anything (need this later to re-annotate)
    #sparse_matrix = sp.sparse.csc_matrix(adata.X)
    #sp.sparse.save_npz(f"results/{tissue}/tables/raw_counts_sparse.npz", sparse_matrix)
    #print("Saved sparse matrix")
    ## Save index and columns
    #with open(f"results/{tissue}/tables/raw_cells.txt", 'w') as file:
    #    for index, row in adata.obs.iterrows():
    #        file.write(str(index) + '\n')
    #
    #print("Saved cells")
    #with open(f"results/{tissue}/tables/raw_genes.txt", 'w') as file:
    #    for index, row in adata.var.iterrows():
    #        file.write(str(index) + '\n')
    #        
    #print("Saved genes")
    #del sparse_matrix
        
    # Also save the gene var df
    adata.var.to_csv(f"results/{tissue}/tables/raw_gene.var.txt", sep = "\t")

    # Discard other inflams if wanted to 
    if discard_other_inflams == "yes":
        adata = adata[adata.obs['disease_status'].isin(['healthy', 'cd'])]
        print(f"Shape of adata after CD/healthy subset = {adata.shape}")

    # Make all blood lineages immune (and categories 'blood')? 
    if all_blood_immune == "yes":
        adata.obs[lineage_column] = adata.obs[lineage_column].astype(str)
        adata.obs.loc[adata.obs['tissue'] == 'blood', lineage_column] = 'Immune'
        adata.obs['category__machine'] = adata.obs['category__machine'].astype(str)
        adata.obs.loc[adata.obs['tissue'] == 'blood', 'category__machine'] = 'blood'


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
    if "None" not in plot_within:
        cats = np.unique(adata.obs['category__machine'].astype(str))
        lins = np.unique(adata.obs[lineage_column])

        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
        for t in tissues:
            data = np.log10(adata.obs[adata.obs.tissue == t].total_counts)
            sns.distplot(data, hist=False, rug=True, label=t)
            #
            plt.legend()
            plt.xlabel('log10(nUMI)')
            plt.title(f"Absolute cut off (black): {min_nUMI}")
            plt.axvline(x = np.log10(min_nUMI), color = 'black', linestyle = '--', alpha = 0.5)
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
            plt.axvline(x = np.log10(min_nUMI), color = 'black', linestyle = '--', alpha = 0.5)
            plt.title(f"{t} - Absolute cut off (black): {min_nUMI}")
            plt.savefig(f"{qc_path}/raw_nUMI_per_category_{t}.png", bbox_inches='tight')
            plt.clf()


        # Plot per lineage, within tissue
        for t in tissues:
            print(t)
            plt.figure(figsize=(8, 6))
            fig,ax = plt.subplots(figsize=(8,6))
            for l in lins:
                data = np.log10(adata.obs[(adata.obs.tissue == t) & (adata.obs[lineage_column] == l)].total_counts)
                absolute_diff = np.abs(data - np.median(data))
                mad = np.median(absolute_diff)
                if nMad_directionality == "uni":
                    cutoff = np.median(data) - (float(relative_nMAD_threshold) * mad)
                    line_color = sns.color_palette("tab10")[int(np.where(lins == l)[0])]
                    sns.distplot(data, hist=False, rug=True, color=line_color, label=f'{l} (relative): {10**cutoff:.2f}')
                    plt.axvline(x = cutoff, linestyle = '--', color = line_color, alpha = 0.5)
                else:
                    cutoff_low = np.median(data) - (float(relative_nMAD_threshold) * mad)
                    cutoff_high = np.median(data) + (float(relative_nMAD_threshold) * mad)
                    line_color = sns.color_palette("tab10")[int(np.where(lins == l)[0])]
                    sns.distplot(data, hist=False, rug=True, color=line_color, label=f'{l} (relative): {10**cutoff_low:.2f}-{10**cutoff_high:.2f}')
                    plt.axvline(x = cutoff_low, linestyle = '--', color = line_color, alpha = 0.5)
                    plt.axvline(x = cutoff_high, linestyle = '--', color = line_color, alpha = 0.5)
            #
            plt.legend()
            plt.xlabel('log10(nUMI)')
            plt.axvline(x = np.log10(min_nUMI), color = 'black', linestyle = '--', alpha = 0.5,  label=f"absolute: {min_nUMI}")
            plt.title(f"{t} - Absolute cut off (black): {min_nUMI}")
            plt.savefig(f"{qc_path}/raw_nUMI_per_lineage_{t}.png", bbox_inches='tight')
            plt.clf()


    # Filter for this cut off if using an absolute cut off, or define a relative cut off or BOTH. NOTE: Can do this based on absolute or relative counts
    adata.obs['log10_total_counts'] = np.log10(adata.obs['total_counts'])
    if use_absolute_nUMI == "yes" and use_relative_mad == "no":
        # adata = adata[adata.obs['total_counts'] > min_nUMI]
        adata.obs['total_counts_keep'] = adata.obs['total_counts'] > min_nUMI

    if use_relative_mad == "yes":
        print("Defining nMAD cut off: total_counts_keep")
        if relative_nUMI_log == "yes":
            adata.obs['total_counts_nMAD'] = nmad_append(adata.obs, 'log10_total_counts', relative_grouping)
        else:
            adata.obs['total_counts_nMAD'] = nmad_append(adata.obs, 'total_counts', relative_grouping)
        
        if nMad_directionality == "uni":
            adata.obs['total_counts_keep'] = adata.obs['total_counts_nMAD'] > -(relative_nMAD_threshold) # Direction aware
        else:
            adata.obs['total_counts_keep'] = abs(adata.obs['total_counts_nMAD']) < relative_nMAD_threshold
        # However, can also apply the min threshold still:
        if use_absolute_nUMI == "yes": 
            adata.obs.loc[adata.obs['total_counts'] < min_nUMI, 'total_counts_keep'] = False
        
        # print the results
        #for t in tissues:
        #    print(f"Min for {t}:")
        #    for l in np.unique(adata.obs[adata.obs['tissue'] == t][lineage_column]):
        #        this_thresh = min(adata.obs[(adata.obs['tissue'] == t) & (adata.obs[lineage_column] == l) & (adata.obs['total_counts_keep'] == True)]['total_counts'])
        #        print(f"{l}: {this_thresh}")
        
        # Plot this results across all cells
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
        data = np.log10(adata.obs.total_counts)
        if nMad_directionality == "uni":
            cutoff = max(adata.obs['log10_total_counts'][adata.obs['total_counts_keep'] == True])
            print(f"The minimum threshold for total_counts is: {cutoff}")
            sns.distplot(data, hist=False, rug=True, label=f'(relative): {10**cutoff:.2f}')
            plt.axvline(x = cutoff, linestyle = '--', alpha = 0.5)
        else:
            cutoff_low = min(adata.obs['log10_total_counts'][adata.obs['total_counts_keep'] == True])
            cutoff_high = max(adata.obs['log10_total_counts'][adata.obs['total_counts_keep'] == True])
            print(f"The window for total counts is: {cutoff_low} - {cutoff_high}")
            sns.distplot(data, hist=False, rug=True, label=f'(relative): {10**cutoff_low:.2f}-{10**cutoff_high:.2f}')
            plt.axvline(x = cutoff_low, linestyle = '--', alpha = 0.5)
            plt.axvline(x = cutoff_high, linestyle = '--', alpha = 0.5)

        plt.legend()
        plt.xlabel('log10(nUMI)')
        plt.axvline(x = np.log10(min_nUMI), color = 'black', linestyle = '--', alpha = 0.5,  label=f"absolute: {min_nUMI}")
        plt.title(f"Absolute cut off (black): {min_nUMI}")
        plt.savefig(f"{qc_path}/raw_nUMI_nmad_thresholded_{tissue}.png", bbox_inches='tight')
        plt.clf()
        
        

    # If filtering sequentially:
    if filter_sequentially == "yes":
        adata = adata[adata.obs['total_counts_keep'] == True]

    # 2. nGene
    print("Filtration of cells with low nGenes")
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    for t in tissues:
        data = np.log10(adata.obs[adata.obs.tissue == t].n_genes_by_counts)
        sns.distplot(data, hist=False, rug=True, label=t)

    plt.legend()
    plt.xlabel('log10(nGene)')
    plt.title(f"Absolute cut off: {min_nGene}")
    plt.axvline(x = np.log10(min_nGene), color = 'red', linestyle = '--', alpha = 0.5)
    plt.savefig(qc_path + '/raw_nGene_per_tissue.png', bbox_inches='tight')
    plt.clf()

    if "None" not in plot_within:
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
            plt.axvline(x = np.log10(min_nGene), color = 'black', linestyle = '--', alpha = 0.5)
            plt.title(f"{t} - Absolute cut off (black): {min_nGene}")
            plt.savefig(f"{qc_path}/raw_nGene_per_category_{t}.png", bbox_inches='tight')
            plt.clf()

        # Plot per lineage, within tissue
        for t in tissues:
            print(t)
            plt.figure(figsize=(8, 6))
            fig,ax = plt.subplots(figsize=(8,6))
            for l in lins:
                data = np.log10(adata.obs[(adata.obs.tissue == t) & (adata.obs[lineage_column] == l)].n_genes_by_counts)
                absolute_diff = np.abs(data - np.median(data))
                mad = np.median(absolute_diff)
                if nMad_directionality == "uni":
                    cutoff = np.median(data) - (float(relative_nMAD_threshold) * mad)
                    line_color = sns.color_palette("tab10")[int(np.where(lins == l)[0])]
                    sns.distplot(data, hist=False, rug=True, color=line_color, label=f'{l} (relative): {10**cutoff:.2f}')
                    plt.axvline(x = cutoff, linestyle = '--', color = line_color, alpha = 0.5)
                else:
                    cutoff_low = np.median(data) - (float(relative_nMAD_threshold) * mad)
                    cutoff_high = np.median(data) + (float(relative_nMAD_threshold) * mad)
                    line_color = sns.color_palette("tab10")[int(np.where(lins == l)[0])]
                    sns.distplot(data, hist=False, rug=True, color=line_color, label=f'{l} (relative): {10**cutoff_low:.2f}-{10**cutoff_high:.2f}')
                    plt.axvline(x = cutoff_low, linestyle = '--', color = line_color, alpha = 0.5)
                    plt.axvline(x = cutoff_high, linestyle = '--', color = line_color, alpha = 0.5)
            
            plt.legend()
            plt.xlabel('log10(nGene)')
            plt.axvline(x = np.log10(min_nGene), color = 'black', linestyle = '--', alpha = 0.5)
            plt.title(f"{t} - Absolute cut off (black): {min_nGene}")
            plt.savefig(f"{qc_path}/raw_nGene_per_lineage_{t}.png", bbox_inches='tight')
            plt.clf()
                
        
    # Filter for this cut off if using an absolute cut off, or define a relative cut off
    adata.obs['log10_n_genes_by_counts'] = np.log10(adata.obs['n_genes_by_counts'])
    if use_absolute_nGene == "yes" and use_relative_mad == "no":
        # adata = adata[adata.obs['total_counts'] > min_nUMI]
        adata.obs['n_genes_by_counts_keep'] = adata.obs['n_genes_by_counts'].astype(int) > min_nGene

    if use_relative_mad == "yes":
        print("Defining nMAD cut off: total_counts_keep")
        if relative_nGene_log == "yes":
            adata.obs['n_genes_by_counts_nMAD'] = nmad_append(adata.obs, 'log10_n_genes_by_counts', relative_grouping)
        else:
            adata.obs['n_genes_by_counts_nMAD'] = nmad_append(adata.obs, 'n_genes_by_counts', relative_grouping)
        
        if nMad_directionality == "uni":
            adata.obs['n_genes_by_counts_keep'] = adata.obs['n_genes_by_counts_nMAD'] > -(relative_nMAD_threshold) # Direction aware
        else:
            adata.obs['n_genes_by_counts_keep'] = abs(adata.obs['n_genes_by_counts_nMAD']) < relative_nMAD_threshold
        
        # However, can also apply the min threshold still:
        if use_absolute_nGene == "yes": 
            adata.obs.loc[adata.obs['n_genes_by_counts'] < min_nGene, 'n_genes_by_counts_keep'] = False
        
        # print the results
        #for t in tissues:
        #    print(f"Min for {t}:")
        #    for l in np.unique(adata.obs[adata.obs['tissue'] == t][lineage_column]):
        #        this_thresh = min(adata.obs[(adata.obs['tissue'] == t) & (adata.obs[lineage_column] == l) & (adata.obs['n_genes_by_counts_keep'] == True)]['n_genes_by_counts'])
        #        print(f"{l}: {this_thresh}")        
        
        # Plot this results across all cells
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
        data = np.log10(adata.obs.n_genes_by_counts)
        if nMad_directionality == "uni":
            cutoff = max(adata.obs['log10_n_genes_by_counts'][adata.obs['n_genes_by_counts_keep'] == True])
            print(f"The minimum threshold for total_counts is: {cutoff}")
            sns.distplot(data, hist=False, rug=True, label=f'(relative): {10**cutoff:.2f}')
            plt.axvline(x = cutoff, linestyle = '--', alpha = 0.5)
        else:
            cutoff_low = min(adata.obs['log10_n_genes_by_counts'][adata.obs['n_genes_by_counts_keep'] == True])
            cutoff_high = max(adata.obs['log10_n_genes_by_counts'][adata.obs['n_genes_by_counts_keep'] == True])
            print(f"The window for nGenes is: {cutoff_low} - {cutoff_high}")
            sns.distplot(data, hist=False, rug=True, label=f'(relative): {10**cutoff_low:.2f}-{10**cutoff_high:.2f}')
            plt.axvline(x = cutoff_low, linestyle = '--', alpha = 0.5)
            plt.axvline(x = cutoff_high, linestyle = '--', alpha = 0.5)

        plt.legend()
        plt.xlabel('log10(nGenes)')
        plt.axvline(x = np.log10(min_nGene), color = 'black', linestyle = '--', alpha = 0.5,  label=f"absolute: {min_nGene}")
        plt.title(f"Absolute cut off (black): {min_nGene}")
        plt.savefig(f"{qc_path}/raw_nGenes_nmad_thresholded_{tissue}.png", bbox_inches='tight')
        plt.clf()

    # If filtering sequentially:
    if filter_sequentially == "yes":
        adata = adata[adata.obs['n_genes_by_counts_keep'] == True]

    # 3. MT% 
    print("Filtration of cells with abnormal/low MT%")
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    for t in tissues:
        data = adata.obs[adata.obs.tissue == t].pct_counts_gene_group__mito_transcript
        sns.distplot(data, hist=False, rug=True, label=t)

    plt.legend()
    plt.xlabel('MT%')
    plt.xlim(0,100)
    plt.axvline(x = MTgut, color = 'green', linestyle = '--', alpha = 0.5)
    plt.axvline(x = MTblood, color = 'red', linestyle = '--', alpha = 0.5)
    plt.savefig(qc_path + '/raw_MT_per_tissue.png', bbox_inches='tight')
    plt.clf()

    if "None" not in plot_within:
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
                plt.title(f"{t} - Absolute cut off: {MTblood}")
            else:
                plt.axvline(x = MTgut, color = 'green', linestyle = '--', alpha = 0.5)
                plt.title(f"{t} - Absolute cut off: {MTgut}")
            
            plt.xlim(0,100)
            plt.savefig(f"{qc_path}/raw_MT_per_category_{t}.png", bbox_inches='tight')
            plt.clf()

        # Plot per lineage, within tissue
        for t in tissues:
            print(t)
            plt.figure(figsize=(8, 6))
            fig,ax = plt.subplots(figsize=(8,6))
            for l in lins:
                data = adata.obs[(adata.obs.tissue == t) & (adata.obs[lineage_column] == l)].pct_counts_gene_group__mito_transcript
                absolute_diff = np.abs(data - np.median(data))
                mad = np.median(absolute_diff)
                if nMad_directionality == "uni":
                    cutoff = np.median(data) + (float(relative_nMAD_threshold) * mad)
                    line_color = sns.color_palette("tab10")[int(np.where(lins == l)[0])]
                    sns.distplot(data, hist=False, rug=True, color=line_color, label=f'{l} (relative): {cutoff:.2f}')
                    plt.axvline(x = cutoff, linestyle = '--', color = line_color, alpha = 0.5)
                else:
                    cutoff_low = np.median(data) - (float(relative_nMAD_threshold) * mad)
                    cutoff_high = np.median(data) + (float(relative_nMAD_threshold) * mad)
                    line_color = sns.color_palette("tab10")[int(np.where(lins == l)[0])]
                    sns.distplot(data, hist=False, rug=True, color=line_color, label=f'{l} (relative): {cutoff_low:.2f}-{cutoff_high:.2f}')
                    plt.axvline(x = cutoff_low, linestyle = '--', color = line_color, alpha = 0.5)
                    plt.axvline(x = cutoff_high, linestyle = '--', color = line_color, alpha = 0.5)
            
            plt.legend()
            plt.xlabel('MT%')
            if t == "blood":
                plt.axvline(x = MTblood, color = 'black', linestyle = '--', alpha = 0.5)
                plt.title(f"{t} - Absolute cut off: {MTblood}")
            else:
                plt.axvline(x = MTgut, color = 'black', linestyle = '--', alpha = 0.5)
                plt.title(f"{t} - Absolute cut off (black): {MTgut}")
                if use_MT_thresh_blood_gut_immune == "yes":
                    plt.axvline(x = MTblood, color = 'red', linestyle = '--', alpha = 0.5,  label=f"absolute cut off immune (red): {MTblood}")
            
            plt.xlim(0,100)
            plt.savefig(f"{qc_path}/raw_MT_per_lineage_{t}.png", bbox_inches='tight')
            plt.clf()

    # Filter on the basis of these thresholds
    if use_absolute_MT == "yes" and use_relative_mad == "no":
        blood_mask = (adata.obs['tissue'] == 'blood') & (adata.obs['pct_counts_gene_group__mito_transcript'] < MTblood)
        gut_tissue_mask = (adata.obs['tissue'] != 'blood') & (adata.obs['pct_counts_gene_group__mito_transcript'] < MTgut)
        adata.obs['MT_perc_keep'] = blood_mask | gut_tissue_mask
    else:
        print("Defining MAD cut off: pct_counts_gene_group__mito_transcript")
        adata.obs['MT_perc_nMads'] = nmad_append(adata.obs, 'pct_counts_gene_group__mito_transcript', relative_grouping)
        
        if nMad_directionality == "uni":
            adata.obs['MT_perc_keep'] = adata.obs['MT_perc_nMads'] < relative_nMAD_threshold # Direction aware
        else:
            adata.obs['MT_perc_keep'] = abs(adata.obs['MT_perc_nMads']) < relative_nMAD_threshold
        
        if use_absolute_MT == "yes":
            # Apply the absolute max still
            adata.obs.loc[adata.obs['pct_counts_gene_group__mito_transcript'] > absolute_max_MT, 'MT_perc_keep'] = False
            # Also apply additional absolute max to immune cells (same as absolute threshold for the blood)
            if use_MT_thresh_blood_gut_immune == "yes":
                adata.obs.loc[(adata.obs[lineage_column] == "Immune") & (adata.obs['pct_counts_gene_group__mito_transcript'] > MTblood), 'MT_perc_keep' ] = False
                
        # Print filters
        #for t in tissues:
        #    print(f"Max for {t}:")
        #    for l in np.unique(adata.obs[adata.obs['tissue'] == t][lineage_column]):
        #        this_thresh = max(adata.obs[(adata.obs['tissue'] == t) & (adata.obs[lineage_column] == l) & (adata.obs['MT_perc_keep'] == True)]['pct_counts_gene_group__mito_transcript'])
        #        print(f"{l}: {this_thresh}")
        
        plt.figure(figsize=(8, 6))
        fig,ax = plt.subplots(figsize=(8,6))
        data = adata.obs.pct_counts_gene_group__mito_transcript
        if nMad_directionality == "uni":
            cutoff = max(adata.obs['pct_counts_gene_group__mito_transcript'][adata.obs['MT_perc_keep'] == True])
            print(f"The minimum threshold for total_counts is: {cutoff}")
            sns.distplot(data, hist=False, rug=True, label=f'(relative): {cutoff:.2f}')
            plt.axvline(x = cutoff, linestyle = '--', alpha = 0.5)
        else:
            cutoff_low = min(adata.obs['pct_counts_gene_group__mito_transcript'][adata.obs['MT_perc_keep'] == True])
            cutoff_high = max(adata.obs['pct_counts_gene_group__mito_transcript'][adata.obs['MT_perc_keep'] == True])
            print(f"The window for pct_counts_gene_group__mito_transcript is: {cutoff_low} - {cutoff_high}")
            sns.distplot(data, hist=False, rug=True, label=f'(relative): {cutoff_low:.2f}-{cutoff_high:.2f}')
            plt.axvline(x = cutoff_low, linestyle = '--', alpha = 0.5)
            plt.axvline(x = cutoff_high, linestyle = '--', alpha = 0.5)

        plt.legend()
        plt.xlabel('pct_counts_gene_group__mito_transcript')
        plt.axvline(x = MTgut, color = 'green', linestyle = '--', alpha = 0.5,  label=f"absolute gut: {MTgut}")
        plt.axvline(x = MTgut, color = 'red', linestyle = '--', alpha = 0.5,  label=f"absolute blood: {MTblood}")
        plt.title(f"Absolute cut off (black): {min_nGene}")
        plt.savefig(f"{qc_path}/raw_pct_counts_gene_group__mito_transcript_nmad_thresholded_{tissue}.png", bbox_inches='tight')
        plt.clf()

    # If filtering sequentially:
    if filter_sequentially == "yes":
        adata = adata[adata.obs['MT_perc_keep'] == True]
            
    
    # Remove on the basis of low MT%
    #adata = adata[adata.obs['pct_counts_gene_group__mito_transcript'] > min_MT]        
    #print(f"Number of cells with MT > min_MT = {adata.shape[0]}")
    
    # If filtering sequentially, filter cells at this point BEFORE the filtering of samples
    if filt_cells_pre_samples == "yes": 
        adata.obs['keep_high_QC'] = adata.obs['total_counts_keep'] & adata.obs['n_genes_by_counts_keep'] & adata.obs['MT_perc_keep']
        adata = adata[adata.obs['keep_high_QC'] == True ]
    
    
    # 4. Remove samples with outlying sequencing depth
    adata.obs['samp_tissue'] = adata.obs['experiment_id'].astype('str') + "_" + adata.obs['tissue'].astype('str')
    samp_data = np.unique(adata.obs.samp_tissue, return_counts=True)
    cells_sample = pd.DataFrame({'sample': samp_data[0], 'Ncells':samp_data[1]})
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

    # Have a look at those cells with high number of cells within each tissue
    high_samps = np.array(cells_sample.loc[cells_sample.Ncells > max_ncells_per_sample, "sample"])
    low_samps = np.array(cells_sample.loc[cells_sample.Ncells < min_ncells_per_sample, "sample"])
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

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(depth_count["Mean_nCounts"], depth_count["n_genes_by_counts"],  c=depth_count["High_cell_sample"], alpha=0.7)
    plt.xlabel('Mean counts / cell')
    plt.ylabel('Mean genes detected / cell')
    plt.savefig(f"{qc_path}/sample_mean_counts_ngenes_all.png", bbox_inches='tight')
    plt.legend()
    plt.clf()
    
    # Same for median
    plt.figure(figsize=(8, 6))
    plt.scatter(depth_count["Median_nCounts"], depth_count["Median_nGene_by_counts"],  c=depth_count["High_cell_sample"], alpha=0.7)
    plt.xlabel('Median counts / cell')
    plt.ylabel('Median ngenes detected / cell')
    plt.savefig(f"{qc_path}/sample_median_counts_ngenes_all.png", bbox_inches='tight')
    plt.clf()

    # Plot the distribution per tissue
    cols_to_plot = ["nCells", "Median_nCounts", "Median_nGene_by_counts", "Median_MT"]
    for c in cols_to_plot:
        print(c)
        plt.figure(figsize=(8, 6))
        for t in tissues:
            samps = adata.obs[adata.obs['tissue'] == t]['samp_tissue']
            data = depth_count[depth_count.index.isin(samps)][c]
            sns.distplot(data, hist=False, rug=True, label=t)

        plt.legend()
        plt.xlabel(f'{c}/sample')
        plt.savefig(f"{qc_path}/sample_dist_{c}.png", bbox_inches='tight')
        plt.clf()
        
    # Also have a look by tissue x disease
    adata.obs['disease_tissue'] = adata.obs['tissue'].astype(str) + "_" + adata.obs['disease_status'].astype(str)
    td = np.unique(adata.obs['disease_tissue'])
    for c in cols_to_plot:
        print(c)
        plt.figure(figsize=(8, 6))
        for t in td:
            samps = adata.obs[adata.obs['disease_tissue'] == t]['samp_tissue']
            data = depth_count[depth_count.index.isin(samps)][c]
            sns.distplot(data, hist=False, rug=True, label=t)
    
        plt.legend()
        plt.xlabel(f'{c}/sample')
        plt.savefig(f"{qc_path}/sample_dist_{c}_tissue_disease.png", bbox_inches='tight')
        plt.clf()
    
    # Within tissue for median
    for t in tissues:
        samps = adata.obs[adata.obs['tissue'] == t]['samp_tissue']
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
        samps = adata.obs[adata.obs['disease_tissue'] == t]['samp_tissue']
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
        samps = adata.obs[adata.obs['disease_tissue'] == t]['samp_tissue']
        plt.figure(figsize=(8, 6))
        plt.scatter(depth_count[depth_count.index.isin(samps)]["nCells"], depth_count[depth_count.index.isin(samps)]["Median_nGene_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["High_cell_sample"], alpha=0.7)
        plt.xlabel('nCells')
        plt.ylabel('Median genes detected / cell')
        if t == "blood":
            plt.axvline(x = min_median_nCount_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
            plt.axhline(y = min_median_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
            plt.title(f"{t} - nCells, min_median_nGene: {min_median_nGene_per_samp_blood}")
        else:
            plt.axvline(x = min_median_nCount_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
            plt.axhline(y = min_median_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
            plt.title(f"{t} - nCells, min_median_nGene: {min_median_nGene_per_samp_gut}")
        
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
        new_column_names = ['keep_nMad_' + col for col in cols_sample_relative_filter]
        boolean_series = depth_count[new_column_names].all(axis=1)
        depth_count['samples_keep'] = boolean_series
        depth_count.to_csv(f"{tabpath}/depth_count_pre_cell_filtration.csv")
        
    if use_abs_per_samp == "yes":
        blood_keep = (depth_count['Median_nCounts'] > min_median_nCount_per_samp_blood) & (depth_count['Median_nGene_by_counts'] > min_median_nGene_per_samp_blood)
        blood_keep = blood_keep & depth_count.index.isin(adata.obs[adata.obs['tissue'] == "blood"]['samp_tissue'])
        blood_keep = blood_keep[blood_keep == True].index
        gut_keep = (depth_count['Median_nCounts'] > min_median_nCount_per_samp_gut) & (depth_count['Median_nGene_by_counts'] > min_median_nGene_per_samp_gut)
        gut_keep = gut_keep & depth_count.index.isin(adata.obs[adata.obs['tissue'] != "blood"]['samp_tissue'])
        gut_keep = gut_keep[gut_keep == True].index
        both_keep = list(np.concatenate([blood_keep, gut_keep]))
        adata.obs['samples_keep'] = adata.obs['samp_tissue'].isin(both_keep)
        lost = depth_count.shape[0]-len(both_keep)
        print(f"The number of samples beng lost using absolute per-samp thresholds is {lost}")  
        
        # Filter
        depth_count.reset_index(inplace=True)
        depth_count.rename(columns={"index": "samp_tissue", "n_genes_by_counts": "mean_n_genes_by_counts"}, inplace=True)
        
        adata.obs.reset_index(inplace=True)
        adata.obs = adata.obs.merge(depth_count, how="left", on="samp_tissue")
        adata.obs.set_index("cell", inplace=True)
        adata.obs['samples_keep'] = adata.obs['keep_nMad_log10_nCells'] & adata.obs['keep_nMad_log10_Median_nCounts'] & adata.obs['keep_nMad_log10_Median_nGene_by_counts']
        print(f"Shape before per-samp filter: {adata.shape}")
        adata = adata[adata.obs['samples_keep'] == True ]
        print(f"Shape after per-samp filter: {adata.shape}")
        columns_to_convert = [col for col in adata.obs.columns if col in depth_count.columns and col != 'High_cell_sample' and col != "samp_tissue"]
        print(columns_to_convert)
        # Convert these columns to float in the first DataFrame
        adata.obs[columns_to_convert] = adata.obs[columns_to_convert].astype(float)
    
    # Plot sankey
    temp = adata.obs[["total_counts_keep", "n_genes_by_counts_keep","MT_perc_keep", "samples_keep"]]
    for col in temp.columns:
        temp[col] = col + "-" + temp[col].astype(str)

    temp['input'] = "input"
    grouped = temp.groupby(["input", "total_counts_keep"])
    input_to_total_counts = grouped.size()#.unstack().div(grouped.size().unstack().sum(axis=1), axis=0).stack()
    grouped = temp.groupby(["total_counts_keep", "n_genes_by_counts_keep"])
    total_counts_to_n_genes = grouped.size()#.unstack().div(grouped.size().unstack().sum(axis=1), axis=0).stack()
    # Group by unique combinations of "n_genes_by_counts_keep" and "MT_perc_keep"
    grouped = temp.groupby(["n_genes_by_counts_keep", "MT_perc_keep"])
    # Calculate the percentage of each value of "n_genes_by_counts_keep" that contributes to each value of "MT_perc_keep"
    n_genes_to_MT_perc = grouped.size()#.unstack().div(grouped.size().unstack().sum(axis=1), axis=0).stack()
    # Group by unique combinations of "MT_perc_keep" and "samples_keep"
    grouped = temp.groupby(["MT_perc_keep", "samples_keep"])
    # Calculate the percentage of each value of "MT_perc_keep" that contributes to each value of "samples_keep"
    MT_perc_to_samples = grouped.size()#.unstack().div(grouped.size().unstack().sum(axis=1), axis=0).stack()
    # Concatenate the three DataFrames vertically
    result = pd.concat([input_to_total_counts, total_counts_to_n_genes, n_genes_to_MT_perc, MT_perc_to_samples]).reset_index()
    result.columns = ['source', 'target', 'value']
    # add denom for these per row before normalisation
    #result['param'] = result['target'].str.split('-').str[0]
    #sum_df = pd.DataFrame({'param': np.unique(result['param'])})
    #sum_df['denom'] = 0
    #sum_df['nTrue'] = 0
    #sum_df['nFalse'] = 0
    #for index, r in sum_df.iterrows():
    #    sum_df.loc[index, "denom"] = result[result['param'] == r['param']].value.sum()
    #    sum_df.loc[index, "nTrue"] = temp[temp[r['param']] == f"{r['param']}-True"].shape[0]
    #    sum_df.loc[index, "nFalse"] = temp[temp[r['param']] == f"{r['param']}-False"].shape[0]



    #print("sum_df")
    #print(sum_df)
    ## Normalise
    #for index, r in result.iterrows():
    #    print(index)
    #    if r['source'] in result['target'].values:
    #            denom = sum_df[sum_df['param'] == r['param']].denom.values[0]
    #            result.loc[index, "value"] = result.loc[index, "value"] / denom
    #                                
    
    print("result")
    print(result)
    sankey = hv.Sankey(result, label='Retention of cells on the basis of different metrics')
    sankey.opts(label_position='left', edge_color='target', node_color='index', cmap='tab20')
    hv.output(fig='png')
    hv.save(sankey, f"{qc_path}/sankey_cell_retention_across_QC.png")
    
    # Apply the min/max ncell thresholds
    keep_samps = depth_count[(depth_count['nCells'] > min_ncells_per_sample) & (depth_count['nCells'] < max_ncells_per_sample)].index
    adata = adata[adata.obs['samp_tissue'].isin(keep_samps)]

    # 5. Apply filters to the cells before expression normalisation
    adata.obs['keep_high_QC'] = adata.obs['total_counts_keep'] & adata.obs['n_genes_by_counts_keep'] & adata.obs['MT_perc_keep'] & adata.obs['samples_keep']
    adata = adata[adata.obs['keep_high_QC'] == True ]

    ####################################
    #### Expression Normalisation ######
    ####################################
    print("~~~~~~~~~~~~ Conducting expression normalisation ~~~~~~~~~~~~~")
    sc.pp.filter_genes(adata, min_cells=5)
    print("Filtered genes")

    # Keep a copy of the raw counts
    adata.layers['counts'] = adata.X.copy()
    print("Copied counts")

    # Calulate the CP10K expression
    sc.pp.normalize_total(adata, target_sum=1e4)
    print("CP10K")

    # Now normalise the data to identify highly variable genes (Same as in Tobi and Monika's paper)
    sc.pp.log1p(adata)
    print("log1p")
    
    # Store this (for later, e.g 005 and plotting)
    adata.layers['log1p_cp10k'] = adata.X.copy()
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
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=int(n_variable_genes), subset=True)
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

    # Scale
    sc.pp.scale(adata, max_value=10)
    print("Scaled")

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
    sc.pl.pca(adata, color="tissue", save="_tissue.png")
    #sc.pl.pca(adata, color="category__machine", save="_category.png")
    #sc.pl.pca(adata, color="input", save="_input.png")
    sc.pl.pca(adata, color=lineage_column, save="_lineage.png")

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

    # Save object ahead of batch correction
    adata.write(objpath + "/adata_PCAd.h5ad")


# Execute
if __name__ == '__main__':
    main()
