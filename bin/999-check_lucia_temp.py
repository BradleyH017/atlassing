import scanpy as sc
import scipy as sp


# IMport anndata
adata = sc.read_h5ad("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/lucia_analysis/blood_vs_ti/data/immune_merged/adata_scanvi_res0.75.h5ad")


outdir="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/lucia_temp"
# Save clusters df
annot = adata.obs[['leiden']]
annot.reset_index(inplace=True)
annot.to_csv(f"{outdir}/clusters.csv", index=False)


# Save cells
with open(f"{outdir}/raw_cells.txt", 'w') as file:
    for index, row in adata.obs.iterrows():
        file.write(str(index) + '\n')


# Save genes
with open(f"{outdir}/raw_genes.txt", 'w') as file:
    for index, row in adata.var.iterrows():
        file.write(str(index) + '\n')


# Save expression
sparse_matrix = sp.sparse.csc_matrix(adata.layers['log1p_cp10k'])
sp.sparse.save_npz(f"{outdir}/log1p_cp10k_sparse.npz", sparse_matrix)

## Run 
# module load ISG/singularity/3.9.0    
# singularity shell -B /lustre -B /software /software/hgi/containers/yascp/yascp.cog.sanger.ac.uk-public-singularity_images-wtsihgi_nf_scrna_qc_6bb6af5-2021-12-23-3270149cf265.sif
# cd /lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined
# python bin/005-scanpy_cluster_validate_resolution-keras.py --tissue "lucia" --sparse_matrix_path lucia_temp/log1p_cp10k_sparse.npz --clusters_df lucia_temp/clusters.csv --genes_f lucia_temp/raw_genes.txt --cells_f lucia_temp/raw_cells.txt --keras_param_file lucia_temp/keras-use_params.txt --number_epoch 25 --batch_size 32 --train_size_fraction 0.67 --output_file lucia_temp/09_04_base


