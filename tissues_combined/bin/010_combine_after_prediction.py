#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-02-15'
__version__ = '0.0.1'

# Load in packages
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import pandas as pd
import argparse
import os
import glob
print("Loaded packages")

# Define the run arguments
def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                QC of tissues together
                """
        )
    
    parser.add_argument(
            '-oh', '--orig_h5ad',
            action='store',
            dest='orig_h5ad',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-hq5', '--highQC_h5ad',
            action='store',
            dest='highQC_h5ad',
            required=True,
            help=''
        )
    
    parser.add_argument(
            '-r2o', '--round2_output',
            action='store',
            dest='round2_output',
            required=True,
            help=''
        )
        
    parser.add_argument(
        '-of', '--output_file',
        action='store',
        dest='output_file',
        help='Basename of output files, assuming output in current working directory.'
    )
    
    return parser.parse_args()


def main():
    inherited_options = parse_options()
    orig_h5ad = inherited_options.orig_h5ad
    highQC_h5ad = inherited_options.highQC_h5ad
    round2_output = inherited_options.round2_output
    output_file = inherited_options.output_file
    
    # Testing
    # orig_h5ad="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/tissues_combined/input/adata_raw_input_all.h5ad"
    # highQC_h5ad="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/results_round3/combined/objects/adata_PCAd_batched_umap.h5ad"
    # round2_output="temp/base_round2-"
    
    
    # Construct the paths for the groups in the round2 output
    fpaths = glob.glob(f"{round2_output}*")
    
    # Load each of these and save the results to a single dataframe of cells + annotations
    want_cols = ['cell', 'probability__celltype__epithelial', 'probability__celltype__immune', 'probability__celltype__mesenchymal', 'round1__predicted_celltype', 'round1__predicted_celltype_probability', 'predicted_celltype', 'predicted_celltype_probability']
    cell_annots = []
    for f in fpaths:
        print(f"Working on {f}")
        temp = sc.read_h5ad(f, backed="r")
        temp.obs.reset_index(inplace=True)
        temp = temp.obs[want_cols]
        group = f.replace(f"{round2_output}", "")
        group = group.replace(".h5ad", "")
        group = group.split("_")[1]
        temp['predicted_celltype'] = temp['predicted_celltype'].apply(lambda x: x.replace('celltype_', group))
        print(temp.head())
        cell_annots.append(temp)
        del temp
    
    # Combine
    annots = pd.concat(cell_annots)
    #annots['predicted_celltype'] = annots['predicted_celltype'].apply(lambda x: x.split('_', 1)[1])
    print(f"Shape of annots is: {annots.shape}")
    
    # Now load the high QC set
    hqc = sc.read_h5ad(highQC_h5ad, backed="r")
    print(f"Shape of High QC data is: {hqc.shape}")
    
    # Subset the annots for the annotations that are kept in the highQC set
    annots = annots[annots['predicted_celltype'].isin(hqc.obs['leiden'])]
    print(f"Number of annotated cells post intersection with high QC clusters is {annots.shape[0]}")
    
    # Delete the high QC set to save mem
    del hqc
    
    # Load the original h5ad
    adata = sc.read_h5ad(orig_h5ad, backed="r")
    print(f"Shape of orig h5ad is: {adata.shape}")
    
    # Subset for cells in annotated
    if "cell" not in adata.obs.columns:
        adata.obs.reset_index(inplace=True)
    
    adata = adata[adata.obs['cell'].isin(annots['cell'])]
    
    # merge
    adata.obs = adata.obs.merge(annots, on="cell", how="left")
    
    # Save final version without filters
    adata.obs.set_index("cell", inplace=True)
    adata.write_h5ad(f"{output_file}.h5ad")
        


# Execute
if __name__ == '__main__':
    main()