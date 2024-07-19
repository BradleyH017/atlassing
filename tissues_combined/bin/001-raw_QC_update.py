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

def dist_plot(adata, column, within=None, relative_threshold=None, absolute=None, out="./", thresholded = False, threshold_method = None, out_suffix = "", relative_directionality = None):
    """
    Plots the distribution and relative threshold cut offs for a value of relative_threshold
    
    Parameters:
    adata: 
        anndata object
    column: 
        column of anndata object to compute relative scores and filtration of. Must be numeric
    within: 
        Grouping variable of anndata.obs. For example, would be "tissue" if scores are to be calculated within groups of adata.obs['tissue']
        If none are desired, use a dummy column that is consistent for all rows of adata.obs
    relative_threshold: 
        Numeric value indicating the number of MAD from the median that threshold should be applied from.
    absolute: 
        Optional. Is there also an absolute threshold to be used?
        This can be a dictionary if different absolute thresholds are to be applied to different values of 'within', each element's name must perfectly match the levels of groups.
    out: 
        Directory for plots to be saved in.
    out_suffix:
        Suffix to add to output file name. Name structure will be f"{out}/{column}_per_{within}{out_suffix}.png"
    thresholded: 
        Has the data already been thresholded? If so, there must be a boolean column called adata.obs[f"{column}_keep"] which indicates whether a cell is kept on the basis of this column
    threshold_method:
        Which threshold method to use. Can be one of "specific" or "outer".
        "specific" will apply thresholds that are specific to the grouping of "within"
        "outer" will apply the outermost thresholds from each of the groupings of "within"
    relative_directionality:
        Specifies directionality of relative thresholds. Only used when actually thresholding
    
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Function to plot and calculate the distribution
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    if within is not None:
        groups = np.unique(adata.obs[within])
        for g in groups:
            tempobs = adata.obs[adata.obs[within] == g]
            data = tempobs[column]
            absolute_diff = np.abs(data - np.median(data))
            mad = np.median(absolute_diff)
            line_color = sns.color_palette("tab10")[int(np.where(groups == g)[0])]
            if thresholded:
                if threshold_method == "specific":
                    # Calculate the threshold actually being used for this group
                    cutoff_low = min(tempobs[column][tempobs[f"{column}_keep"] == True])
                    cutoff_high = max(tempobs[column][tempobs[f"{column}_keep"] == True])
                else:
                    # Calculate the threshold actually being used across all groups
                    cutoff_low = min(adata.obs[column][adata.obs[f"{column}_keep"] == True])
                    cutoff_high = max(adata.obs[column][adata.obs[f"{column}_keep"] == True])
            else:
                # Calculate the relative threshold being used for this group of data
                cutoff_low = np.median(data) - (float(relative_threshold) * mad)
                cutoff_high = np.median(data) + (float(relative_threshold) * mad)
            if relative_threshold is not None:
                sns.distplot(data, hist=False, color = line_color, rug=True, label=f'{g} (relative): {cutoff_low:.2f}-{cutoff_high:.2f}')
                plt.axvline(x = cutoff_low, linestyle = '--', color = line_color, alpha = 0.5)
                plt.axvline(x = cutoff_high, linestyle = '--', color = line_color, alpha = 0.5)
            else:
                sns.distplot(data, hist=False, color = line_color, rug=True, label=f'{g}')
    #
    else:
        data = adata.obs[column]
        absolute_diff = np.abs(data - np.median(data))
        mad = np.median(absolute_diff)
        line_color = sns.color_palette("tab10")[int(np.where(groups == g)[0])]
        if thresholded:
            # Calculate the threshold actually being used
            cutoff_low = min(adata.obs[column][adata.obs[f"{column}_keep"] == True])
            cutoff_high = max(adata.obs[column][adata.obs[f"{column}_keep"] == True])
        else:
            # Calculate the relative threshold being used for this group of data
            cutoff_low = np.median(data) - (float(relative_threshold) * mad)
            cutoff_high = np.median(data) + (float(relative_threshold) * mad)
        if relative_threshold is not None:
            sns.distplot(data, hist=False, rug=True, label=f'relative: {cutoff_low:.2f}-{cutoff_high:.2f}')
        else:
            sns.distplot(data, hist=False, rug=True)
    plt.legend()
    plt.xlabel(column)
    if absolute is not None:
        if isinstance(absolute, dict):
            temp = [value for value in absolute.values() if isinstance(value, (int, float))]
            temp = np.unique(temp)
            plt.title(f"Absolute (black): {absolute}. Threshold_method = {threshold_method}, directionality = {relative_directionality}")
        else:
            temp = [absolute]
            plt.title(f"Absolute (black): {absolute:.2f}, Threshold_method = {threshold_method}, directionality = {relative_directionality}")
        for a in temp:
            plt.axvline(x = a, color = 'black', linestyle = '--', alpha = 0.5)
    #
    else:
        plt.title(f"No absolute cut off")
    plt.savefig(f"{out}/{column}_per_{within}{out_suffix}.png", bbox_inches='tight')
    plt.clf()

#adata.obs['log_n_genes_by_counts'] = np.log10(adata.obs['n_genes_by_counts'])
#adata.obs['log_total_counts'] = np.log10(adata.obs['total_counts'])
#cols = ["pct_counts_gene_group__mito_transcript", "log_n_genes_by_counts", "log_total_counts"]
#thresholds = {"pct_counts_gene_group__mito_transcript": {"blood": 20, "r": 50, "ti": 50}, "log_n_genes_by_counts": np.log10(250), "log_total_counts": np.log10(500)}
#for index, c in enumerate(cols):
#    dist_plot(adata, c, within="tissue", relative_threshold=2.5, absolute=thresholds[c], out="results")
#    dist_plot(adata, c, within="manual_lineage", relative_threshold=2.5, absolute=thresholds[c], out="results")

def update_obs_qc_plot_thresh(adata, column = None, within = None, relative_threshold = None, threshold_method = "specific", relative_directionality = "bi", absolute = None, absolute_directionality = "over", plot = True, out = "./", out_suffix = ""):
    """
    Updates the .obs of an anndata object with median absolute deviation scores and plots proposed filters based on these
    
    Parameters:
    adata: 
        anndata object
    column: 
        column of anndata object to compute relative scores and filtration of. Must be numeric
    within: 
        Grouping variable of anndata.obs. For example, would be "tissue" if scores are to be calculated within groups of adata.obs['tissue']
        If none are desired, use a dummy column that is consistent for all rows of adata.obs
    relative_threshold: 
        Numeric value indicating the number of MAD from the median that threshold should be applied from.
    threshold_method: 
        Which threshold method to use. Can be one of "specific", "outer" or None
        "specific" will apply thresholds that are specific to the grouping of "within"
        "outer" will apply the outermost thresholds from each of the groupings of "within".
        None will not apply the relative threshold after calculation
        If "outer" is used in conjunction with unidirectionality (over/under), the max/min threshold will be applied in that direction only
    relative_directionality: 
        What is the directionality of the relative threshold to be used? Can be one of "bi", "over" or "under" for bidirectional, unidirectional keep cells over or unidirectional keep cells under respectively
    absolute: 
        Optional. Is there also an absolute threshold to be used? 
        This can be a dictionary if different absolute thresholds are to be applied to different values of 'within', each element's name must perfectly match the levels of groups.
    absolute_directionality: 
        What is the directionality of cells to keep relative to the absolute threshold? Can be "over" or "under".
    plot: 
        Plot the results?
    out: 
        Directory for plots to be saved in.
    out_suffix:
        Suffix to add to output file name. Name structure will be f"{out}/{column}_per_{within}{out_suffix}.png"
    """
    print(f"----- Calculating thresholds for: {column} ------")
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Calculate the relative threshold per value of within
    adata.obs[f"{column}_nMAD"] = nmad_append(adata.obs, column, within)
    if threshold_method == "specific":
        print(f"Applying {within}-specific nMAD thresholds")
        # Add boolean vector specific to each group
        if relative_directionality == "bi":
            adata.obs[f"{column}_keep"] = abs(adata.obs[f"{column}_nMAD"]) < relative_threshold # Bidirectional
        if relative_directionality == "over":
            adata.obs[f"{column}_keep"] = adata.obs[f"{column}_nMAD"] > relative_threshold
        if relative_directionality == "under":
            adata.obs[f"{column}_keep"] = adata.obs[f"{column}_nMAD"] < relative_threshold
    # determine how thresholds should be applied
    if threshold_method == "outer":
        print(f"Applying outer window of {within}-specific nMAD thresholds")
        # Generate a vector of thresholds
        thresh = []
        groups = np.unique(adata.obs[within])
        for g in groups:
            tempobs = adata.obs[adata.obs[within] == g]
            data = tempobs[column]
            absolute_diff = np.abs(data - np.median(data))
            mad = np.median(absolute_diff)
            cutoff_low = np.median(data) - (float(relative_threshold) * mad)
            cutoff_high = np.median(data) + (float(relative_threshold) * mad)
            thresh.append([cutoff_low, cutoff_high])
        # Flatten this list to find max / min
        flattened_list = [item for sublist in thresh for item in sublist]
        max_value = max(flattened_list)
        min_value = min(flattened_list)
        if relative_directionality == "bi":
            adata.obs[f"{column}_keep"] = (adata.obs[column] > min_value) & (adata.obs[column] < max_value) # Bidirectional
        if relative_directionality == "over":
            adata.obs[f"{column}_keep"] = adata.obs[column] > min_value
        if relative_directionality == "under":
            adata.obs[f"{column}_keep"] = adata.obs[column] < max_value
    if threshold_method == None:
        # Don't apply any relative threshold after calculating. Make dummy for all TRUE
        adata.obs[f"{column}_keep"] = True
    # Add absolute threshold if using
    if absolute is not None:
        # If applying multiple absolute thresholds to each level of within separately
        if isinstance(absolute, dict):
            print(f"Applying {within}-specific absolute thresholds")
            if absolute_directionality == "over":
                for g in absolute.keys():
                    adata.obs.loc[(adata.obs[within] == g) & (adata.obs[column] < absolute[g]), f"{column}_keep"] = False
            if absolute_directionality == "under":
                for g in absolute.keys():
                    adata.obs.loc[(adata.obs[within] == g) & (adata.obs[column] > absolute[g]), f"{column}_keep"] = False
        else:
            print(f"Not applying {within}-specific absolute thresholds")
            if absolute_directionality == "over":
                adata.obs.loc[adata.obs[column] < absolute, f"{column}_keep"] = False
            if absolute_directionality == "under":
                adata.obs.loc[adata.obs[column] > absolute, f"{column}_keep"] = False
    if plot:
        print(f"Plotting to: {out}/{column}_per_{within}{out_suffix}.png")
        # Plot the distribution with the thresholds actually being used (NOTE: This will be dependent on relative grouping)
        dist_plot(adata, column, within=within, relative_threshold=relative_threshold, absolute=absolute, out=out, thresholded = True, threshold_method = threshold_method, out_suffix = out_suffix, relative_directionality = relative_directionality)

#cols = ["pct_counts_gene_group__mito_transcript", "log_n_genes_by_counts", "log_total_counts"]
#thresholds = {"pct_counts_gene_group__mito_transcript": {"blood": 20, "r": 50, "ti": 50}, "log_n_genes_by_counts": np.log10(250), "log_total_counts": np.log10(500)}
#over_under = {"pct_counts_gene_group__mito_transcript": "under", "log_n_genes_by_counts": "over", "log_total_counts": "over"}
#for index, c in enumerate(cols):
#    update_obs_qc_plot_thresh(adata, c, within="tissue", relative_threshold=2.5, threshold_method = "outer", relative_directionality = "bi", absolute=thresholds[c], absolute_directionality = over_under[c], plot =True, out="results", out_suffix = "_THRESHOLDED")
#    update_obs_qc_plot_thresh(adata, c, within="manual_lineage", relative_threshold=2.5, threshold_method = "outer", relative_directionality = "bi", absolute=thresholds[c], absolute_directionality = over_under[c], plot =True, out="results", out_suffix = "_THRESHOLDED")



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
    tissues = np.unique(adata.obs['tissue'])
    for t in tissues:
        tempt = adata.obs[adata.obs['tissue'] == t]
        print(f"For {t}, there is {len(np.unique(tempt['sanger_sample_id']))} samples. A total of {tempt.shape[0]} cells")
        cd = tempt[tempt['disease_status'] == "CD"]
        print(f"{len(np.unique(cd['sanger_sample_id']))} are CD")
        
    # Also save the gene var df
    adata.var.to_csv(f"results/{tissue}/tables/raw_gene.var.txt", sep = "\t")

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
    
    # Plot prelimeinary exploratory plots
    # These are per-grouping for each parameter - to highlight where shared/common QC could be useful
    # Also force this to be done per tissue
    for index, c in enumerate(cols):
        dist_plot(adata, c, within="tissue", relative_threshold=relative_nMAD_threshold, absolute=thresholds[c], out=qc_path)
        dist_plot(adata, c, within=relative_grouping, relative_threshold=relative_nMAD_threshold, absolute=thresholds[c], out=qc_path)
    
    # Now calculate the QC metrics, apply absolute/relative thresholds with the desired method, and save plots of where the thresholds lie based on this grouping
    for index, c in enumerate(cols):
        update_obs_qc_plot_thresh(adata, c, within=relative_grouping, relative_threshold=relative_nMAD_threshold, threshold_method = threshold_method, relative_directionality = "bi", absolute=thresholds[c], absolute_directionality = over_under[c], plot =True, out=qc_path, out_suffix = "_THRESHOLDED")

    # Extract list of high and low cell number samples - PRE QC
    adata.obs['samp_tissue'] = adata.obs['sanger_sample_id'].astype('str') + "_" + adata.obs['tissue'].astype('str')
    samp_data = np.unique(adata.obs.samp_tissue, return_counts=True)
    cells_sample = pd.DataFrame({'sample': samp_data[0], 'Ncells':samp_data[1]})
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
        #
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
    depth_count.to_csv(f"{tabpath}/depth_count_pre_cell_filtration.csv", index=False)
    
    # bind to anndata
    adata.obs.reset_index(inplace=True)
    adata.obs = adata.obs.merge(depth_count.drop(columns="disease_tissue"), how="left", on="samp_tissue")
    columns_to_convert = [col for col in adata.obs.columns if col in depth_count.columns and col != 'High_cell_sample' and col != "samp_tissue" and col != "samples_keep" and col != "disease_tissue"]
    print(f"converting: {columns_to_convert}")
    adata.obs[columns_to_convert] = adata.obs[columns_to_convert].astype(float)
    adata.obs.set_index("cell", inplace=True)
    
    # Plot sankey
    temp = adata.obs[["log_total_counts_keep", "log_n_genes_by_counts_keep", "pct_counts_gene_group__mito_transcript_keep", "samples_keep"]]
    for col in temp.columns:
        temp[col] = col + "-" + temp[col].astype(str)

    temp['input'] = "input"
    grouped = temp.groupby(["input", "log_total_counts_keep"])
    input_to_total_counts = grouped.size()#.unstack().div(grouped.size().unstack().sum(axis=1), axis=0).stack()
    grouped = temp.groupby(["log_total_counts_keep", "log_n_genes_by_counts_keep"])
    total_counts_to_n_genes = grouped.size()#.unstack().div(grouped.size().unstack().sum(axis=1), axis=0).stack()
    # Group by unique combinations of "log_n_genes_by_counts_keep" and "pct_counts_gene_group__mito_transcript_keep"
    grouped = temp.groupby(["log_n_genes_by_counts_keep", "pct_counts_gene_group__mito_transcript_keep"])
    # Calculate the percentage of each value of "log_n_genes_by_counts_keep" that contributes to each value of "pct_counts_gene_group__mito_transcript_keep"
    n_genes_to_MT_perc = grouped.size()#.unstack().div(grouped.size().unstack().sum(axis=1), axis=0).stack()
    # Group by unique combinations of "pct_counts_gene_group__mito_transcript_keep" and "samples_keep"
    grouped = temp.groupby(["pct_counts_gene_group__mito_transcript_keep", "samples_keep"])
    # Calculate the percentage of each value of "pct_counts_gene_group__mito_transcript_keep" that contributes to each value of "samples_keep"
    MT_perc_to_samples = grouped.size()#.unstack().div(grouped.size().unstack().sum(axis=1), axis=0).stack()
    # Concatenate the three DataFrames vertically
    result = pd.concat([input_to_total_counts, total_counts_to_n_genes, n_genes_to_MT_perc, MT_perc_to_samples]).reset_index()
    result.columns = ['source', 'target', 'value']                               
    
    print("result")
    print(result)
    sankey = hv.Sankey(result, label='Retention of cells on the basis of different metrics')
    sankey.opts(label_position='left', edge_color='target', node_color='index', cmap='tab20')
    hv.output(fig='png')
    hv.save(sankey, f"{qc_path}/sankey_cell_retention_across_QC.png")
    
    # Apply sample-level threshold
    adata = adata[adata.obs['samples_keep'] == True ]
    
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
     tempt = adata.obs[adata.obs['tissue'] == t]
     print(f"For {t}, there is {len(np.unique(tempt['sanger_sample_id']))} samples. A total of {tempt.shape[0]} cells")
     cd = tempt[tempt['disease_status'] == "CD"]
     print(f"{len(np.unique(cd['sanger_sample_id']))} are CD")

    ####################################
    ######### PCA calculation ##########
    ####################################

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

    # Save object ahead of batch correction
    adata.write(objpath + "/adata_PCAd.h5ad")


# Execute
if __name__ == '__main__':
    main()
