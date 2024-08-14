# Tobi August 2024
# Generate a celltypist model from an adata object

import pandas as pd
import numpy as np
import scanpy as sc
import celltypist
import argparse
import os

def main():
    """Run CLI."""
    parser = argparse.ArgumentParser(
        description="""
            Trains a celltypist model from a h5ad file
            """
    )
    parser.add_argument(
        '-h5', '--h5_anndata',
        action='store',
        dest='h5',
        required=True,
        help='Path to h5 AnnData file.'
    )
    
    parser.add_argument(
        '-o', '--outdir',
        action='store',
        dest='outdir',
        default='.',
        required=False,
        help='Output directory'
    )    
    
    parser.add_argument(
        '-a', '--annotation',
        action='store',
        dest='annotation',
        default=None,
        required=True,
        help='Name of the adata.obs column containing labels for model training'
    )   
    
    parser.add_argument(
        '-vge', '--variable_genes_exclude',
        action='store',
        dest='vge',
        default='',
        help='Tab-delimited file with genes to exclude from the highly\
            variable gene list. Must contain ensembl_gene_id column.\
            (default: None - keep all variable genes)'
    )

    parser.add_argument(
        '--gene_symbols',
        action='store_true',
        default=False,
        dest='gene_symbols',
        help='Pulls gene names from adata.var.gene_symbols'
    )
    
    options = parser.parse_args()

    # Load the AnnData file.
    print('Loading AnnData')
    adata = sc.read_h5ad(filename=options.h5)
    print(f"Shape of anndata is {adata.shape}")

    if options.vge != '':
        exclude_hv_gene_df = pd.read_csv(options.vge, sep='\t')

        # Exclude genes from highly variable gene set.
        if len(exclude_hv_gene_df) > 0:
            print(f'Anndata shape before excluding genes: {np.shape(adata)}')
            adata = adata[:,~adata.var.index.isin(exclude_hv_gene_df['ensembl_gene_id'])]
            print(f'Anndata shape after excluding genes: {np.shape(adata)}')

    if not "log1p_cp10k" in adata.layers.keys():
        print('Normalising data')
        sc.pp.normalize_total(
                    adata,
                    target_sum=1e4,
                    exclude_highly_expressed=False,
                    key_added='normalization_factor',  # add to adata.obs
                    inplace=True)
        sc.pp.log1p(adata)
    else:
        adata.X = adata.layers['log1p_cp10k']

    if options.gene_symbols:
        print('Using gene symbols')
        adata.var['ENS'] = adata.var_names
        adata.var_names = list(adata.var['gene_symbols'])
        adata.var_names_make_unique()

    #Training a CellTypist model.
    print('Starting training')
    new_model = celltypist.train(adata, labels = options.annotation, n_jobs=-1, feature_selection = True)
    print('Finishing training')
    #Write out the model.
    outname = os.path.splitext(os.path.basename(options.h5))[0]
    new_model.write(f'{options.outdir}/{options.annotation}-{outname}.pkl')
    print('Finished saving model')


if __name__ == '__main__':
    main()