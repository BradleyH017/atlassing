#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-01-12'
__version__ = '0.0.1'

# Load libraries
import scanpy as sc
import rapids_singlecell as rsc
print("Loaded libraries")


# Parse script options (ref, query, outdir)
def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                Batch correction of tissues together
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
        '-bc', '--batch_correction',
        action='store',
        dest='batch_correction',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-col', '--batch_column',
        action='store',
        dest='batch_column',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-sacol', '--scANVI_col',
        action='store',
        dest='scANVI_col',
        required=False,
        help=''
    )
    
    parser.add_argument(
        '-cvo', '--correct_variable_only',
        action='store',
        dest='correct_variable_only',
        required=False,
        help=''
    )

    return parser.parse_args()

def main():
    # Parse options
    inherited_options = parse_options()
    input_file = inherited_options.input_file
    batch_column=inherited_options.batch_column
    correct_variable_only=inherited_options.correct_variable_only
    
    # TESTING
    # tissue="rectum"
    # input_file="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/alternative_results/bi_nmads_results/rectum/objects/adata_PCAd.h5ad"
    # batch_column="samp_tissue"
    # correct_variable_only=True
    
    # Derive and print the tissue arguments
    tissue=inherited_options.tissue
    print(f"~~~~~~~ TISSUE:{tissue}")
    
    # Load in anndata
    adata = sc.read_h5ad(input_file)
    
    # Get basedir - modified to run in current dir
    # outdir = os.path.dirname(os.path.commonprefix([input_file]))
    # outdir = os.path.dirname(os.path.commonprefix([outdir]))

    print("Set up outdirs")
    
    if correct_variable_only:
            adata = adata[:,adata.var['highly_variable'] == True].copy()

    print("~~~~~~~~~~~~~~~~~~~ Batch correcting with Harmony ~~~~~~~~~~~~~~~~~~~")
    rsc.pp.harmony_integrate(adata, batch_column, basis='X_pca', adjusted_basis='X_Harmony')
    sparse_matrix = sp.sparse.csc_matrix(adata.obsm['X_Harmony'])
    sp.sparse.save_npz(f"results/{tissue}/tables/batch_correction/Harmony_matrix.npz", sparse_matrix)
    

# Execute
if __name__ == '__main__':
    main()