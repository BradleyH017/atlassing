#!/usr/bin/env python
__version__ = '0.0.3'

# Taken from yascp: https://github.com/wtsi-hgi/yascp/blob/1d465f848a700b9d39c4c9a63356bd7b985f5a1c/bin/0057-predict_clusters_keras_model-anndata.py
# Module: https://github.com/wtsi-hgi/yascp/blob/1d465f848a700b9d39c4c9a63356bd7b985f5a1c/modules/nf-core/modules/keras_celltype/main.nf
# Adjust by Bradley Feb 2024 so that input can be matrices, not a h5ad, as cannot combine new h5ad for scvi batch correction with keras that runs on gpu
# This script will use the scaled log1p_cp10k matrix from all cells, comparable to approach when clusters tested with keras

import argparse
import os
import random
import warnings
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
import anndata as ad
# import csv
# from distutils.version import LooseVersion

import keras
from sklearn import preprocessing

import plotnine as plt9
import matplotlib
matplotlib.use('Agg')

# Compression level for anndata. 9 = highest but slow
compression_level = 9


# Set seed for reproducibility
seed_value = 0
# 0. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 1. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 2. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 3. Set the `tensorflow` pseudo-random generator at a fixed value
# tf.random.set_seed(seed_value)


def filter_genes(adata, df_genes_exclude):
    """Filters genes from andata matrix.
    df_genes_exclude should have either an ensembl_gene_id column or
    gene_symbol column.
    """
    for col in df_genes_exclude.columns:
        filt = np.isin(adata.var.index, df_genes_exclude[col].values)
        if np.any(filt):
            print('Removing {} genes'.format(filt.sum()))
            filt = np.invert(filt)
            foo = adata[:, filt]
            foo_shape = foo.X.shape[1]
            for layer in foo.layers:
                print(layer)
                if foo.layers[layer].shape[1] != foo_shape:
                    raise Exception('Layers not updated for filter')
            adata = foo
    return(adata)


def label_mito_ribo(adata):
    """Labels mito and ribo genes."""
    # Label mitochondrial encoded trancripts
    # This includes:
    # * Mitochondrially encoded protein coding genes
    # * Mitochondrially encoded transcribed regions
    # * Mitochondrially encoded RNAs
    #
    # The below gene list was downloaded from on 3 Aug 2020:
    # https://www.genenames.org/data/genegroup/#!/group/1972
    gene_group__mito_transcript = [
        'MT-7SDNA',
        'MT-ATP6',
        'MT-ATP8',
        'MT-ATT',
        'MT-CO1',
        'MT-CO2',
        'MT-CO3',
        'MT-CSB1',
        'MT-CSB2',
        'MT-CSB3',
        'MT-CYB',
        'MT-HPR',
        'MT-HSP1',
        'MT-HSP2',
        'MT-LIPCAR',
        'MT-LSP',
        'MT-ND1',
        'MT-ND2',
        'MT-ND3',
        'MT-ND4',
        'MT-ND4L',
        'MT-ND5',
        'MT-ND6',
        'MT-OHR',
        'MT-OLR',
        'MT-RNR1',
        'MT-RNR2',
        'MT-RNR3',
        'MT-TA',
        'MT-TAS',
        'MT-TC',
        'MT-TD',
        'MT-TE',
        'MT-TER',
        'MT-TF',
        'MT-TFH',
        'MT-TFL',
        'MT-TFX',
        'MT-TFY',
        'MT-TG',
        'MT-TH',
        'MT-TI',
        'MT-TK',
        'MT-TL1',
        'MT-TL2',
        'MT-TM',
        'MT-TN',
        'MT-TP',
        'MT-TQ',
        'MT-TR',
        'MT-TS1',
        'MT-TS2',
        'MT-TT',
        'MT-TV',
        'MT-TW',
        'MT-TY'
    ]
    adata.var['gene_group__mito_transcript'] = [
        x in gene_group__mito_transcript for x in adata.var['gene_symbols']
    ]
    # mito_gene_list = sc.queries.mitochondrial_genes() # another query
    # adata.var['gene_group__mito_transcript'] = [
    #     x.startswith('MT-') for x in adata.var['gene_symbols']
    # ]
    # use this if var_names='gene_symbols' in sc.read_10x_mtx
    # adata.var['mito_gene'] = [
    #     x.startswith('MT-') for x in adata.var_names
    # ]
    # Label mitochondrial encoded proteins
    #
    # The below gene list was downloaded from on 3 Aug 2020:
    # https://www.genenames.org/data/genegroup/#!/group/1974
    gene_group__mito_protein = [
        'MT-ATP6',
        'MT-ATP8',
        'MT-CO1',
        'MT-CO2',
        'MT-CO3',
        'MT-CYB',
        'MT-ND1',
        'MT-ND2',
        'MT-ND3',
        'MT-ND4',
        'MT-ND4L',
        'MT-ND5',
        'MT-ND6'
    ]
    adata.var['gene_group__mito_protein'] = [
        x in gene_group__mito_protein for x in adata.var['gene_symbols']
    ]
    # Label ribosomal protein genes
    # Ribosomal protein: A ribosomal protein is any of
    # the proteins that, in conjunction with rRNA, make up the ribosomal
    # subunits involved in the cellular process of translation. A large
    # part of the knowledge about these organic molecules has come from the
    # study of E. coli ribosomes. Most ribosomal proteins have been
    # isolated and specific antibodies have been produced. These, together
    # with electronic microscopy and the use of certain reactives, have
    # allowed for the determination of the topography of the proteins in
    # ribosome. E.coli, other bacteria and Archaea have a 30S small subunit
    # and a 50S large subunit, whereas humans and yeasts have a 40S small
    # subunit and a 60S large subunit. Equivalent subunits are frequently
    # numbered differently between bacteria, Archaea, yeasts and humans.
    #
    # The below gene list was downloaded from on 3 Aug 2020:
    # https://www.genenames.org/data/genegroup/#!/group/1054
    gene_group__ribo_protein = set([
        'DAP3',
        'FAU',
        'MRPL1',
        'MRPL1',
        'MRPL10',
        'MRPL10',
        'MRPL11',
        'MRPL11',
        'MRPL12',
        'MRPL12',
        'MRPL13',
        'MRPL13',
        'MRPL14',
        'MRPL14',
        'MRPL15',
        'MRPL15',
        'MRPL16',
        'MRPL16',
        'MRPL17',
        'MRPL17',
        'MRPL18',
        'MRPL18',
        'MRPL19',
        'MRPL19',
        'MRPL2',
        'MRPL2',
        'MRPL20',
        'MRPL20',
        'MRPL21',
        'MRPL21',
        'MRPL22',
        'MRPL22',
        'MRPL23',
        'MRPL23',
        'MRPL24',
        'MRPL24',
        'MRPL27',
        'MRPL27',
        'MRPL28',
        'MRPL28',
        'MRPL3',
        'MRPL3',
        'MRPL30',
        'MRPL30',
        'MRPL32',
        'MRPL32',
        'MRPL33',
        'MRPL33',
        'MRPL34',
        'MRPL34',
        'MRPL35',
        'MRPL35',
        'MRPL36',
        'MRPL36',
        'MRPL37',
        'MRPL37',
        'MRPL38',
        'MRPL38',
        'MRPL39',
        'MRPL39',
        'MRPL4',
        'MRPL4',
        'MRPL40',
        'MRPL40',
        'MRPL41',
        'MRPL41',
        'MRPL42',
        'MRPL42',
        'MRPL43',
        'MRPL43',
        'MRPL44',
        'MRPL44',
        'MRPL45',
        'MRPL45',
        'MRPL46',
        'MRPL46',
        'MRPL47',
        'MRPL47',
        'MRPL48',
        'MRPL48',
        'MRPL49',
        'MRPL49',
        'MRPL50',
        'MRPL50',
        'MRPL51',
        'MRPL51',
        'MRPL52',
        'MRPL52',
        'MRPL53',
        'MRPL53',
        'MRPL54',
        'MRPL54',
        'MRPL55',
        'MRPL55',
        'MRPL57',
        'MRPL57',
        'MRPL58',
        'MRPL9',
        'MRPS10',
        'MRPS10',
        'MRPS11',
        'MRPS11',
        'MRPS12',
        'MRPS12',
        'MRPS14',
        'MRPS14',
        'MRPS15',
        'MRPS15',
        'MRPS16',
        'MRPS16',
        'MRPS17',
        'MRPS17',
        'MRPS18A',
        'MRPS18A',
        'MRPS18B',
        'MRPS18B',
        'MRPS18C',
        'MRPS18C',
        'MRPS2',
        'MRPS2',
        'MRPS21',
        'MRPS21',
        'MRPS22',
        'MRPS22',
        'MRPS23',
        'MRPS23',
        'MRPS24',
        'MRPS24',
        'MRPS25',
        'MRPS25',
        'MRPS26',
        'MRPS26',
        'MRPS27',
        'MRPS27',
        'MRPS28',
        'MRPS28',
        'MRPS30',
        'MRPS30',
        'MRPS31',
        'MRPS31',
        'MRPS33',
        'MRPS33',
        'MRPS34',
        'MRPS34',
        'MRPS35',
        'MRPS35',
        'MRPS36',
        'MRPS36',
        'MRPS5',
        'MRPS6',
        'MRPS7',
        'MRPS9',
        'RPL10',
        'RPL10A',
        'RPL10L',
        'RPL11',
        'RPL12',
        'RPL13A',
        'RPL14',
        'RPL15',
        'RPL17',
        'RPL18A',
        'RPL19',
        'RPL21',
        'RPL22',
        'RPL23',
        'RPL23A',
        'RPL24',
        'RPL26',
        'RPL26L1',
        'RPL27',
        'RPL27A',
        'RPL28',
        'RPL29',
        'RPL3',
        'RPL30',
        'RPL31',
        'RPL32',
        'RPL34',
        'RPL35',
        'RPL35A',
        'RPL36',
        'RPL36A',
        'RPL36AL',
        'RPL37',
        'RPL37A',
        'RPL38',
        'RPL39',
        'RPL39L',
        'RPL3L',
        'RPL4',
        'RPL41',
        'RPL5',
        'RPL6',
        'RPL7',
        'RPL7A',
        'RPL7L1',
        'RPL8',
        'RPL9',
        'RPLP0',
        'RPLP1',
        'RPLP2',
        'RPS10',
        'RPS11',
        'RPS12',
        'RPS13',
        'RPS14',
        'RPS15',
        'RPS15A',
        'RPS16',
        'RPS17',
        'RPS18',
        'RPS19',
        'RPS2',
        'RPS20',
        'RPS21',
        'RPS23',
        'RPS24',
        'RPS25',
        'RPS26',
        'RPS27',
        'RPS27A',
        'RPS27L',
        'RPS28',
        'RPS29',
        'RPS3',
        'RPS3A',
        'RPS4X',
        'RPS4Y1',
        'RPS4Y2',
        'RPS5',
        'RPS6',
        'RPS7',
        'RPS8',
        'RPS9',
        'UBA52'
    ])
    adata.var['gene_group__ribo_protein'] = [
        x in gene_group__ribo_protein for x in adata.var['gene_symbols']
    ]
    # Label ribosomal RNA
    #
    # The below gene list was downloaded from on 3 Aug 2020:
    # https://www.genenames.org/data/genegroup/#!/group/848
    gene_group__ribo_rna = [
        'MT-RNR1',
        'MT-RNR2',
        'RNA18S1',
        'RNA18S2',
        'RNA18S3',
        'RNA18S4',
        'RNA18S5',
        'RNA18SN1',
        'RNA18SN2',
        'RNA18SN3',
        'RNA18SN4',
        'RNA18SN5',
        'RNA28S1',
        'RNA28S2',
        'RNA28S3',
        'RNA28S4',
        'RNA28S5',
        'RNA28SN1',
        'RNA28SN2',
        'RNA28SN3',
        'RNA28SN4',
        'RNA28SN5',
        'RNA45S1',
        'RNA45S2',
        'RNA45S3',
        'RNA45S4',
        'RNA45S5',
        'RNA45SN1',
        'RNA45SN2',
        'RNA45SN3',
        'RNA45SN4',
        'RNA45SN5',
        'RNA5-8S1',
        'RNA5-8S2',
        'RNA5-8S3',
        'RNA5-8S4',
        'RNA5-8S5',
        'RNA5-8SN1',
        'RNA5-8SN2',
        'RNA5-8SN3',
        'RNA5-8SN4',
        'RNA5-8SN5',
        'RNA5S1',
        'RNA5S10',
        'RNA5S11',
        'RNA5S12',
        'RNA5S13',
        'RNA5S14',
        'RNA5S15',
        'RNA5S16',
        'RNA5S17',
        'RNA5S2',
        'RNA5S3',
        'RNA5S4',
        'RNA5S5',
        'RNA5S6',
        'RNA5S7',
        'RNA5S8',
        'RNA5S9',
        'RNR1',
        'RNR2',
        'RNR3',
        'RNR4',
        'RNR5'
    ]
    adata.var['gene_group__ribo_rna'] = [
        x in gene_group__ribo_rna for x in adata.var['gene_symbols']
    ]
    return(adata)


def comma_labels(x_list):
    """Change list of int to comma format."""
    result = []
    for x in x_list:
        result.append(format(int(x), ','))
    return(result)


def plot_kept_cells(
    adata,
    label_column='cluster',
    filter_column='predicted_celltype_probability',
    keep_greater_than=[0.75, 0.8, 0.9, 0.99],
    out_file_base='andata',
    title='Number of cells',
    scale_log10=False,
    ratio=False
):
    """Plot adata."""
    df_list = []
    df_all = adata.obs.groupby(
        [label_column]
    ).size().reset_index(name='n_cells')
    df_all['type'] = 'All cells'
    df_all['filter_value'] = np.nan
    df_list.append(df_all)
    for x in keep_greater_than:
        adata_tmp = adata[adata.obs[filter_column] > x, :]
        df_tmp = adata_tmp.obs.groupby(
            [label_column]
        ).size().reset_index(name='n_cells')
        df_tmp['type'] = '{} >{}'.format(
            filter_column.capitalize().replace('_', ' '),
            str(x)
        )
        df_tmp['filter_value'] = x
        df_list.append(df_tmp)
    df_plot = pd.concat(df_list)
    df_all = df_all.set_index(label_column)
    df_plot['fraction_of_cells'] = (
        df_plot['n_cells'] /
        df_all.loc[df_plot[label_column], 'n_cells'].values
    )
    # Make barplots of the data
    if not ratio:
        gplt = plt9.ggplot(df_plot, plt9.aes(
            x=label_column,
            y='n_cells',
            fill='type'
        ))
        gplt = gplt + plt9.labs(
            title=title,
            x='Cell type',
            y='Number of cells',
            fill='Filter'
        )
        gplt = gplt + plt9.geom_bar(stat='identity', position='dodge')
        out_name = '{}-kept_number_cells-barplot.png'.format(
            out_file_base,
        )
        if scale_log10:
            gplt = gplt + plt9.scale_y_continuous(
                trans='log10',
                labels=comma_labels
            )
            out_name = '{}-kept_number_cells_log10-barplot.png'.format(
                out_file_base,
            )
    else:
        df_plot_tmp = df_plot.loc[df_plot['type'] != 'All cells', :]
        gplt = plt9.ggplot(df_plot_tmp, plt9.aes(
            x=label_column,
            y='fraction_of_cells',
            color='type'
        ))
        gplt = gplt + plt9.labs(
            title=title,
            x='Cell type',
            y='Fraction of cells kept',
            color='Filter'
        )
        gplt = gplt + plt9.geom_point(alpha=0.75)
        gplt = gplt + plt9.ylim(0, 1)
        out_name = '{}-kept_number_cells-scatter.png'.format(
            out_file_base,
        )
    gplt = gplt + plt9.theme_bw()
    gplt = gplt + plt9.theme(
        axis_text_x=plt9.element_text(angle=-90, hjust=0.5)
    )
    gplt.save(
        out_name,
        # dpi=300,
        width=12,
        height=8,
        limitsize=False
    )
    return(gplt)


def plot_number_cells(
    adata,
    label_column='predicted_celltype',
    out_file_base='andata',
    title='Number of cells',
    scale_log10=False
):
    """Plot adata.
    Parameters
    ----------
    adata : Anndata object
        Description of parameter `adata`.
    Returns
    -------
    Nothing
    """
    df_plt = adata.obs.groupby(
        [label_column]
    ).size().reset_index(name='n_cells')
    gplt = plt9.ggplot(df_plt, plt9.aes(
        x=label_column,
        y='n_cells'
    ))
    gplt = gplt + plt9.theme_bw()
    gplt = gplt + plt9.geom_bar(stat='identity', position='dodge')
    gplt = gplt + plt9.labs(
        title=title,
        x='Cell type',
        y='Number of cells'
    )
    gplt = gplt + plt9.theme(
        axis_text_x=plt9.element_text(angle=-45, hjust=0)
    )
    out_name = '{}-number_cells-barplot.png'.format(
        out_file_base,
    )
    if scale_log10:
        gplt = gplt + plt9.scale_y_continuous(
            trans='log10',
            labels=comma_labels
        )
        out_name = '{}-number_cells_log10-barplot.png'.format(
            out_file_base,
        )
    gplt.save(
        out_name,
        # dpi=300,
        width=12,
        height=8,
        limitsize=False
    )


def plot_old_vs_new_cell_labels(
    adata,
    old_label='cluster',
    new_label='predicted_celltype',
    plot_type='n_cells',
    out_file_base='andata'
):
    """Plot adata.
    Parameters
    ----------
    adata : Anndata object
        Description of parameter `adata`.
    type : string
        Valid options ['n_cells', 'avg_prob']
    Returns
    -------
    Nothing
    """
    # Figure out what dataframe we need to make
    if plot_type == 'n_cells':
        df1 = adata.obs.groupby(
            [old_label, new_label]
        ).size().reset_index(name='nr_cells_cluster_cell_type')
        df2 = adata.obs.groupby(
            [old_label]
        ).size().reset_index(name='nr_cells_cluster')
        df_plt = pd.merge(df1, df2, on=old_label)
        df_plt['frac_cells_cluster_cell_type'] = (
            df_plt['nr_cells_cluster_cell_type'] / df_plt['nr_cells_cluster']
        )
        # Plot breakdown of cells and original clusters in the filtered data
        gplt = plt9.ggplot(df_plt, plt9.aes(
            x=old_label,
            y=new_label
        ))
        gplt = gplt + plt9.geom_point(
            plt9.aes(
                size='frac_cells_cluster_cell_type',
                color='nr_cells_cluster_cell_type',
                alpha='frac_cells_cluster_cell_type'
            )
        )
        gplt = gplt + plt9.labs(
            title='Predicted cell type (top prediction per cell)',
            x='Original cell type',
            y='Predicted cell type',
            color='Number of cells',
            size='Fraction of cells (predicted/original)',
            alpha='Fraction of cells (predicted/original)'
        )
    elif plot_type == 'avg_prob':
        df_prediction_classes_mean = adata.obs.groupby(
            [old_label, new_label]
        ).agg({'predicted_celltype_probability': 'mean'}).reset_index()
        df_prediction_classes_mean = df_prediction_classes_mean.rename(
            columns={'predicted_celltype_probability': 'mean_probability'}
        )
        gplt = plt9.ggplot(df_prediction_classes_mean, plt9.aes(
            x=old_label,
            y=new_label
        ))
        gplt = gplt + plt9.theme_bw()
        gplt = gplt + plt9.geom_point(
            plt9.aes(
                size='mean_probability',
                color='mean_probability',
                alpha='mean_probability'
            )
        )
        gplt = gplt + plt9.labs(
            title='Average cell type probablity {}'.format(
                'across all cells'
            ),
            x='Original cell cluster',
            y='Predicted cell type',
            size='Average probability',
            color='Average probability',
            alpha='Average probability'
        )
    else:
        raise Exception('Invalid plot_type')
    gplt = gplt + plt9.theme_bw()
    gplt = gplt + plt9.theme(
        axis_text_x=plt9.element_text(angle=-90, hjust=0.5)
    )
    gplt.save(
        '{}-prediction_comparison-dotplot-{}.png'.format(
            out_file_base,
            plot_type
        ),
        # dpi=300,
        width=10,
        height=10,
        limitsize=False
    )


def main():
    """Run CLI."""
    parser = argparse.ArgumentParser(
        description="""
            Predicts cell types of a new dataset using old labels.
            """
    )

    parser.add_argument(
        '-t', '--tissue',
        action='store',
        dest='tissue',
        required=True,
        help='tissue being ran on'
    )
    
    parser.add_argument(
        '-h5', '--h5ad',
        action='store',
        dest='h5ad',
        required=True,
        help='path h5ad for the entire data file'
    )
    
    parser.add_argument(
        '-m', '--model',
        action='store',
        dest='model',
        required=True,
        help='path to the model'
    )
    
    parser.add_argument(
        '-w', '--weights',
        action='store',
        dest='weights',
        required=True,
        help='path to the model weights'
    )

    parser.add_argument(
        '-of', '--output_file',
        action='store',
        dest='output_file',
        help='Basename of output files, assuming output in current working directory.'
    )
    
    options = parser.parse_args()
    tissue = options.tissue
    h5ad = options.h5ad
    model = options.optim_resolution_f
    weights = options.weights
    out_file_base = options.output_file
    print(f"out_file_base: {out_file_base}")

    # For development
    # tissue="combined"
    #h5ad="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/tissues_combined/input/adata_raw_input_all.h5ad"
    #model="results_round2/combined/tables/lineage_model/base.h5"
    #weights="results_round2/combined/tables/lineage_model/base-weights.tsv.gz"
    #out_file_base="temp/base-"

    # Load the anndata object
    adata = sc.read_h5ad(h5ad)
    if "cell" in adata.obs.columns:
        adata.obs.set_index("cell", inplace=True)

    # Compute log1p if not already
    if not "log1p_cp10k" in adata.layers.keys():
        sc.pp.normalize_total(adata, target_sum=1e4)
        print("CP10K")
        sc.pp.log1p(adata)
        print("log1p")
        adata.layers['log1p_cp10k'] = adata.X.copy()
    else:
        adata.X = adata.layers['log1p_cp10k']
        
    # Delete automatically added uns - UPDATE: bad idea to delete as this slot
    # is used in _highly_variable_genes_single_batch.
    # del adata.uns['log1p']
    # Add record of this operation.
    # adata.layers['log1p_cpm'] = adata.X.copy()
    # adata.uns['log1p_cpm'] = {'transformation': 'ln(CPM+1)'}
    adata.uns['log1p_cp10k'] = {'transformation': 'ln(CP10k+1)'}

            
            
    # We assume anndata.var.index corresponds to ensembl gene ids
    if 'ensembl_gene_id' not in adata.var:
        warnings.warn(
            'Assming andata.var.index corresponds to ensembl gene ids.'
        )
    else:
        # Set the index of anndata to ensembl_gene_id. If there
        # are duplicate ensembl_gene_id, then fix them.
        adata.var['ensembl_gene_id_unique'] = adata.var[
            'ensembl_gene_id'
        ].astype(str).copy()
        filt = adata.var['ensembl_gene_id_unique'].astype(str).duplicated()
        if np.any(filt):
            warnings.warn(
                'Found duplicate ensembl gene ids: {}'.format(
                    ','.join(adata.var['ensembl_gene_id'][filt])
                )
            )
            warnings.warn(
                'Setting duplicates to original index of gene: {}'.format(
                    ','.join(adata.var.index[filt])
                )
            )
            adata.var['ensembl_gene_id_unique'][filt] = adata.var.index[
                filt
            ].values
        adata.var['original_index'] = adata.var.index.copy()
        adata.var.set_index('ensembl_gene_id_unique', inplace=True)
        if np.any(adata.var.index.duplicated()):
            warnings.warn(
                'Still found duplicate ensembl gene ids.\
                    Running var_names_make_unique()'
            )
            adata.var_names_make_unique()


    # Load the model and weights:
    model = keras.models.load_model(model)
    import gzip
    with gzip.open(weights, 'rt') as compressed_file:
        # Use pandas read_csv to load the data
        df_weights = pd.read_csv(weights, sep = "\t")

    # Extract the counts we are interested in
    X = adata.layers['log1p_cp10k'].copy()

    # Center and scale the data - like we did in the script were we trained
    # the keras model
    if sp.sparse.issparse(X):
        X = X.todense()
    # Subset and expand X to use the same genes as the input model. If
    # the gene is missing, then it will be assumed to have 0 counts.
    all_same = np.array_equal(
        adata.var.index,
        df_weights['ensembl_gene_id'].values
    )
    if not all_same:
        df_X = pd.DataFrame(X)
        df_X.index = adata.obs.index
        df_X.columns = adata.var.index
        # Get the columns that we need that are missing and fill them in
        # with 0s (TODO: test NA)
        missing_cols = np.setdiff1d(
            df_weights['ensembl_gene_id'].values,
            df_X.columns
        )
        if len(missing_cols) == 0:
            # then the issue is not missing genes, but just a different order
            # of genes
            df_X_fixed = df_X
        else:
            warnings.warn(
                'Setting expression values for {} missing genes to 0.0'.format(
                    len(missing_cols)
                )
            )
            # NOTE: if we get an error below it is likely because the columns
            # are not all unique.
            df_X_fixed = df_X.reindex(
                columns=[*df_X.columns.tolist(), *missing_cols],
                fill_value=0.0
            )
        # for i in missing_cols:
        #     df_X[i] = 0.0  # np.nan will not work
        # Re-order the ensembl ids to fit the model
        df_X_fixed = df_X_fixed[df_weights['ensembl_gene_id'].values]
        X = df_X_fixed.values

    # Scale
    scaler = preprocessing.StandardScaler(
        with_mean=True,
        with_std=True
    )
    X_std = scaler.fit_transform(X)

    # Celltype probabilities ##################################################
    # Predict the labels of each cell using the model
    prediction_classes = model.predict(X_std)
    # NOTE: model.predict_proba same as model.predict
    # prediction_classes_proba = model.predict_proba(X_std)

    # Make a dataframe of the cell id and their prediction matrix
    df_prediction_classes = pd.DataFrame(prediction_classes)
    df_prediction_classes.index = adata.obs.index
    # NOTE: the mappings of predictions cols follow the df_weights order,
    # THESE NUMBERS DO NOT NECISSARILY CORRESPOND TO CLUSTERS
    df_prediction_classes.columns = [
        'celltype__{}'.format(i) for i in df_weights.columns[1:]
    ]

    # Clean up the final names
    df_prediction_classes.columns = [
        'probability__{}'.format(i) for i in df_prediction_classes.columns
    ]

    # Annotate the top hit
    df_top_prediction = pd.DataFrame({
            'predicted_celltype': df_prediction_classes.idxmax(axis=1),
            'predicted_celltype_probability': df_prediction_classes.max(axis=1)
        })

    df_top_prediction['predicted_celltype'] = df_top_prediction[
        'predicted_celltype'
    ].str.replace('probability__', '')

    df_prediction_classes = df_prediction_classes.merge(df_top_prediction, left_index=True, right_index=True)

    # Save the h5ad of these results
    import os
    directory_path = os.path.dirname(out_file_base)
    print(f"out_file_base: {out_file_base}")
    print(f"directory_path: {directory_path}")
    if os.path.exists(directory_path) == False:
        os.mkdir(directory_path)

    # MERGE THIS WITH THE H5AD
    adata.obs.reset_index(inplace=True)
    df_prediction_classes.reset_index(inplace=True)
    adata.obs = adata.obs.merge(df_prediction_classes, on="cell", how="left")
    adata.write_h5ad(f"{out_file_base}.h5ad")


if __name__ == '__main__':
    main()