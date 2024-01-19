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
import argparse
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
            temp.set_index("cell", inplace=True)
        
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
        '-lineage_column', '--lineage_column',
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
        '-samp_nCount_blood', '--min_mean_nCount_per_samp_blood',
        action='store',
        dest='min_mean_nCount_per_samp_blood',
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
    input_file = inherited_options.input_file
    outdir = inherited_options.outdir
    discard_other_inflams = inherited_options.discard_other_inflams
    relative_grouping = inherited_options.relative_grouping
    relative_grouping = relative_grouping.split("|")
    all_blood_immune = inherited_options.all_blood_immune
    min_nUMI = float(inherited_options.min_nUMI)
    use_absolute_nUMI = inherited_options.use_absolute_nUMI
    use_relative_mad = inherited_options.use_relative_mad
    lineage_column = inherited_options.lineage_column
    relative_nMAD_threshold = float(inherited_options.relative_nMAD_threshold)
    relative_nUMI_log = inherited_options.relative_nUMI_log
    min_nGene = float(inherited_options.min_nGene)
    use_absolute_nGene = inherited_options.use_absolute_nGene
    relative_nGene_log = inherited_options.relative_nGene_log
    MTgut = float(inherited_options.MT_thresh_gut)
    MTblood = float(inherited_options.MT_thresh_blood)
    use_absolute_MT = inherited_options.use_absolute_MT
    absolute_max_MT = float(inherited_options.absolute_max_MT)
    min_mean_nCount_per_samp_blood = float(inherited_options.min_mean_nCount_per_samp_blood)
    min_mean_nCount_per_samp_gut = float(inherited_options.min_mean_nCount_per_samp_gut)
    min_mean_nGene_per_samp_blood = float(inherited_options.min_mean_nGene_per_samp_blood)
    min_mean_nGene_per_samp_gut = float(inherited_options.min_mean_nGene_per_samp_gut)
    use_abs_per_samp = inherited_options.use_abs_per_samp
    filt_blood_keras = inherited_options.filt_blood_keras
    n_variable_genes = float(inherited_options.n_variable_genes)
    remove_problem_genes = inherited_options.remove_problem_genes
    
    print("~~~~~~~~~ Running arguments ~~~~~~~~~")
    print(f"input_file: {input_file}")
    print(f"discard_other_inflams:{discard_other_inflams}")
    print(f"all_blood_immune:{all_blood_immune}")
    print(f"min_nUMI:{min_nUMI}")
    print(f"use_absolute_nUMI:{use_absolute_nUMI}")
    print(f"relative_grouping:{relative_grouping}")
    print(f"relative_nMAD_threshold:{relative_nMAD_threshold}")
    print(f"relative_nUMI_log:{relative_nUMI_log}")
    print(f"min_nGene:{min_nGene}")
    print(f"use_absolute_nGene:{use_absolute_nGene}")
    print(f"relative_nGene_log:{relative_nGene_log}")
    print(f"MTgut:{MTgut}")
    print(f"MTblood:{MTblood}")
    print(f"use_absolute_MT:{use_absolute_MT}")
    print(f"absolute_max_MT:{absolute_max_MT}")
    print(f"min_mean_nCount_per_samp_blood:{min_mean_nCount_per_samp_blood}")
    print(f"min_mean_nCount_per_samp_gut:{min_mean_nCount_per_samp_gut}")
    print(f"min_mean_nGene_per_samp_blood:{min_mean_nGene_per_samp_blood}")
    print(f"min_mean_nGene_per_samp_gut:{min_mean_nGene_per_samp_gut}")
    print(f"use_abs_per_samp:{use_abs_per_samp}")
    print(f"filt_blood_keras:{filt_blood_keras}")
    print(f"n_variable_genes:{n_variable_genes}")
    print(f"remove_problem_genes:{remove_problem_genes}")
    print("Parsed args")

    # Get basedir
    outdir = os.path.dirname(os.path.commonprefix([input_file]))
    outdir = os.path.dirname(os.path.commonprefix([outdir]))
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

    # Discard other inflams if wanted to 
    if discard_other_inflams == "yes":
        adata = adata[adata.obs['disease_status'].isin(['healthy', 'cd'])]

    # Make all blood lineages immune (and categories 'blood')? 
    if all_blood_immune == "yes":
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
    cats = np.unique(adata.obs['category__machine'].astype(str))
    lins = np.unique(adata.obs[lineage_column])

    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    for t in tissues:
        data = np.log10(adata.obs[adata.obs.tissue == t].total_counts)
        sns.distplot(data, hist=False, rug=True, label=t)

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
            cutoff = np.median(data) - (relative_nMAD_threshold * mad)
            line_color = sns.color_palette("tab10")[int(np.where(lins == l)[0])]
            sns.distplot(data, hist=False, rug=True, color=line_color, label=f'{l} (relative): {10**cutoff:.2f}')
            plt.axvline(x = cutoff, linestyle = '--', color = line_color, alpha = 0.5)
        
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
        adata.obs['total_counts_keep'] = adata[adata.obs['total_counts'] > min_nUMI]

    if use_relative_mad == "yes":
        print("Defining nMAD cut off: total_counts_keep")
        if relative_nUMI_log == "yes":
            adata.obs['total_counts_nMAD'] = nmad_append(adata.obs, 'log10_total_counts', relative_grouping)
        else:
            adata.obs['total_counts_nMAD'] = nmad_append(adata.obs, 'total_counts', relative_grouping)
        
        adata.obs['total_counts_keep'] = adata.obs['total_counts_nMAD'] > -(relative_nMAD_threshold) # Direction aware
        # However, can also apply the min threshold still:
        if use_absolute_nUMI == "yes": 
            adata.obs.loc[adata.obs['total_counts'] < min_nUMI, 'total_counts_keep'] = False
        
        # print the results
        for t in tissues:
            print(f"Min for {t}:")
            for l in np.unique(adata.obs[adata.obs['tissue'] == t][lineage_column]):
                this_thresh = min(adata.obs[(adata.obs['tissue'] == t) & (adata.obs[lineage_column] == l) & (adata.obs['total_counts_keep'] == True)]['total_counts'])
                print(f"{l}: {this_thresh}")


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
            cutoff = np.median(data) - (relative_nMAD_threshold * mad)
            line_color = sns.color_palette("tab10")[int(np.where(lins == l)[0])]
            sns.distplot(data, hist=False, rug=True, color=line_color, label=f'{l} (relative): {10**cutoff:.2f}')
            plt.axvline(x = cutoff, linestyle = '--', color = line_color, alpha = 0.5)
        
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
        adata.obs['n_genes_by_counts_keep'] = adata[adata.obs['n_genes_by_counts'].astype(int) > min_nGene]

    if use_relative_mad == "yes":
        print("Defining nMAD cut off: total_counts_keep")
        if relative_nGene_log == "yes":
            adata.obs['n_genes_by_counts_nMAD'] = nmad_append(adata.obs, 'log10_n_genes_by_counts', relative_grouping)
        else:
            adata.obs['n_genes_by_counts_nMAD'] = nmad_append(adata.obs, 'n_genes_by_counts', relative_grouping)
        
        adata.obs['n_genes_by_counts_keep'] = adata.obs['n_genes_by_counts_nMAD'] > -(relative_nMAD_threshold) # Direction aware
        # However, can also apply the min threshold still:
        if use_absolute_nGene == "yes": 
            adata.obs.loc[adata.obs['n_genes_by_counts'] < min_nGene, 'n_genes_by_counts_keep'] = False
        
        # print the results
        for t in tissues:
            print(f"Min for {t}:")
            for l in np.unique(adata.obs[adata.obs['tissue'] == t][lineage_column]):
                this_thresh = min(adata.obs[(adata.obs['tissue'] == t) & (adata.obs[lineage_column] == l) & (adata.obs['n_genes_by_counts_keep'] == True)]['n_genes_by_counts'])
                print(f"{l}: {this_thresh}")        
    

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
            cutoff = np.median(data) + (relative_nMAD_threshold * mad)
            line_color = sns.color_palette("tab10")[int(np.where(lins == l)[0])]
            sns.distplot(data, hist=False, rug=True, color=line_color, label=f'{l} (relative): {cutoff:.2f}')
            plt.axvline(x = cutoff, linestyle = '--', color = line_color, alpha = 0.5)
        
        plt.legend()
        plt.xlabel('MT%')
        if t == "blood":
            plt.axvline(x = MTblood, color = 'black', linestyle = '--', alpha = 0.5)
            plt.title(f"{t} - Absolute cut off: {MTblood}")
        else:
            plt.axvline(x = MTgut, color = 'black', linestyle = '--', alpha = 0.5)
            plt.title(f"{t} - Absolute cut off (black): {MTgut}")
        
        plt.xlim(0,100)
        plt.savefig(f"{qc_path}/raw_MT_per_lineage_{t}.png", bbox_inches='tight')
        plt.clf()

    # Filter on the basis of these thresholds
    if use_absolute_MT == "yes" and use_relative_mad == "no":
        blood_mask = (adata.obs['tissue'] == 'blood') & (adata.obs['pct_counts_gene_group__mito_transcript'] < MTblood)
        gut_tissue_mask = (adata.obs['tissue'] != 'blood') & (adata.obs['pct_counts_gene_group__mito_transcript'] < MTgut)
        adata['MT_perc_keep'] = blood_mask | gut_tissue_mask
    else:
        print("Defining MAD cut off: pct_counts_gene_group__mito_transcript")
        adata.obs['MT_perc_nMads'] = nmad_append(adata.obs, 'pct_counts_gene_group__mito_transcript', relative_grouping)
        adata.obs['MT_perc_keep'] = adata.obs['MT_perc_nMads'] < relative_nMAD_threshold # Direction aware
        if use_absolute_MT == "yes":
            # Apply the absolute max still
            adata.obs.loc[adata.obs['pct_counts_gene_group__mito_transcript'] > absolute_max_MT, 'MT_perc_keep'] = False
        # Print filters
        for t in tissues:
            print(f"Max for {t}:")
            for l in np.unique(adata.obs[adata.obs['tissue'] == t][lineage_column]):
                this_thresh = max(adata.obs[(adata.obs['tissue'] == t) & (adata.obs[lineage_column] == l) & (adata.obs['MT_perc_keep'] == True)]['pct_counts_gene_group__mito_transcript'])
                print(f"{l}: {this_thresh}")

            
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
    high_samps = np.array(cells_sample.loc[cells_sample.Ncells > 10000, "sample"])
    low_samps = np.array(cells_sample.loc[cells_sample.Ncells < 500, "sample"])
    # Summarise
    depth_count = pd.DataFrame(index = np.unique(adata.obs.samp_tissue), columns=["Mean_nCounts", "nCells", "High_cell_sample", "n_genes_by_counts"])
    for s in range(0, depth_count.shape[0]):
        samp = depth_count.index[s]
        depth_count.iloc[s,1] = adata.obs[adata.obs.samp_tissue == samp].shape[0]
        depth_count.iloc[s,0] = sum(adata.obs[adata.obs.samp_tissue == samp].total_counts)/depth_count.iloc[s,1]
        depth_count.iloc[s,3] = sum(adata.obs[adata.obs.samp_tissue == samp].n_genes_by_counts)/depth_count.iloc[s,1]
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
    plt.savefig(f"{qc_path}/sample_mean_counts_ngenes_all.png", bbox_inches='tight')
    plt.clf()

    # Plot the distribution per tissue
    plt.figure(figsize=(8, 6))
    for t in tissues:
        samps = adata.obs[adata.obs['tissue'] == t]['samp_tissue']
        data = depth_count[depth_count.index.isin(samps)]["Mean_nCounts"]
        sns.distplot(data, hist=False, rug=True, label=t)

    plt.legend()
    plt.xlabel('Mean nCounts/cell')
    plt.savefig(qc_path + '/sample_dist_mean_counts.png', bbox_inches='tight')
    plt.clf()

    for t in tissues:
        samps = adata.obs[adata.obs['tissue'] == t]['samp_tissue']
        data = depth_count[depth_count.index.isin(samps)]["n_genes_by_counts"]
        sns.distplot(data, hist=False, rug=True, label=t)

    plt.legend()
    plt.xlabel('Mean nCounts/cell')
    plt.savefig(qc_path + '/sample_dist_mean_nGenes.png', bbox_inches='tight')
    plt.clf()

    # Within tissue
    for t in tissues:
        samps = adata.obs[adata.obs['tissue'] == t]['samp_tissue']
        plt.figure(figsize=(8, 6))
        plt.scatter(depth_count[depth_count.index.isin(samps)]["Mean_nCounts"], depth_count[depth_count.index.isin(samps)]["n_genes_by_counts"],  c=depth_count[depth_count.index.isin(samps)]["High_cell_sample"], alpha=0.7)
        plt.xlabel('Mean counts / cell')
        plt.ylabel('Mean genes detected / cell')
        if t == "blood":
            plt.axvline(x = min_mean_nCount_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
            plt.axhline(y = min_mean_nGene_per_samp_blood, color = 'red', linestyle = '--', alpha = 0.5)
            plt.title(f"{t} - min_mean_nCount: {min_mean_nCount_per_samp_blood}, min_mean_nGene: {min_mean_nGene_per_samp_blood}")
        else:
            plt.axvline(x = min_mean_nCount_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
            plt.axhline(y = min_mean_nGene_per_samp_gut, color = 'red', linestyle = '--', alpha = 0.5)
            plt.title(f"{t} - min_mean_nCount: {min_mean_nCount_per_samp_gut}, min_mean_nGene: {min_mean_nGene_per_samp_gut}")
        
        plt.savefig(f"{qc_path}/sample_mean_counts_ngenes_{t}.png", bbox_inches='tight')
        plt.clf()

    # These are clearly different across tissues
    # Either apply absolute thresholds or a relative cut off
    if use_abs_per_samp == "yes" and use_relative_mad == "no":
        blood_keep = (depth_count['Mean_nCounts'] > min_mean_nCount_per_samp_blood) & (depth_count['n_genes_by_counts'] > min_mean_nGene_per_samp_blood)
        blood_keep = blood_keep & depth_count.index.isin(adata.obs[adata.obs['tissue'] == "blood"]['experiment_id'])
        blood_keep = blood_keep[blood_keep == True].index
        gut_keep = (depth_count['Mean_nCounts'] > min_mean_nCount_per_samp_gut) & (depth_count['n_genes_by_counts'] > min_mean_nGene_per_samp_gut)
        gut_keep = gut_keep & depth_count.index.isin(adata.obs[adata.obs['tissue'] != "blood"]['experiment_id'])
        gut_keep = gut_keep[gut_keep == True].index
        both_keep = list(np.concatenate([blood_keep, gut_keep]))
        adata['samples_keep'] = adata.obs['samp_tissue'].isin(both_keep)
    else:
        # Calculating relative threshold
        print("Calculating relative threshold: sample level mean nCells and depth")
        sample_tissue = adata.obs[['samp_tissue','tissue']].reset_index()
        sample_tissue = sample_tissue[['samp_tissue','tissue']].drop_duplicates()
        depth_count.reset_index(inplace=True)
        depth_count = depth_count.rename(columns = {"index": "samp_tissue"})
        depth_count = depth_count.merge(sample_tissue, on="samp_tissue")
        depth_count.set_index("samp_tissue", inplace=True)
        nMads=depth_count.groupby('tissue')[['Mean_nCounts', 'n_genes_by_counts']].apply(nmad_calc)
        unique_tissues = nMads.index.get_level_values('tissue').unique()
        result_list = []
        for tissue in unique_tissues:
            tissue_data = pd.DataFrame(nMads.loc[tissue])
            tissue_data['tissue'] = tissue
            result_list.append(tissue_data)
            #
        result_df = pd.concat(result_list)
        result_df = result_df.reset_index()
        result_df = result_df.rename(columns = {'Mean_nCounts': 'Mean_nCounts_nMad', 'n_genes_by_counts': 'n_genes_by_counts_nMad'})
        depth_count = depth_count.merge(result_df[['samp_tissue', 'Mean_nCounts_nMad', 'n_genes_by_counts_nMad']], on="samp_tissue")
        # Exclude those on the lower end only
        depth_count['samp_Mean_nCounts_keep'] = depth_count['Mean_nCounts_nMad'] > -(relative_nMAD_threshold) # Direction aware
        depth_count['samp_n_genes_by_counts_keep'] = depth_count['n_genes_by_counts_nMad'] > -(relative_nMAD_threshold) # Direction aware
        depth_count['keep_both'] = depth_count['samp_Mean_nCounts_keep'] & depth_count['samp_n_genes_by_counts_keep']
        adata.obs['samples_keep'] = adata.obs['samp_tissue'].isin(depth_count[depth_count['keep_both'] == True]['samp_tissue'])
        # print filters
        print("For sample level samp_Mean_nCounts")
        for t in tissues:
            print(f"for {t}:")
            this_thresh = min(depth_count[(depth_count['tissue'] == t) & (depth_count['keep_both'] == True)]['Mean_nCounts'])
            print(this_thresh)
        #
        print("For sample level samp_n_genes_by_counts")
        for t in tissues:
            print(f"for {t}:")
            this_thresh = min(depth_count[(depth_count['tissue'] == t) & (depth_count['keep_both'] == True)]['n_genes_by_counts'])
            print(this_thresh)

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

    # identify highly variable genes and scale these ahead of PCA
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=int(n_variable_genes))
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
    sc.pl.pca(adata, color="category__machine", save="_category.png")
    sc.pl.pca(adata, color="input", save="_input.png")
    sc.pl.pca(adata, color=lineage_column, save="_lineage.png")

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
    loadings.to_csv(tabpath + "/PCA_loadings.csv")

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
        labels_key=lineage_column,
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
    
