###### Temp script to re-annotate the lineage of cells

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy as sp

dir="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/alternative_results/bi_nmads_results"
tissues=["blood", "ti", "rectum", "gut", "all"]
gene_lists={"epithelial": ["EPCAM", "KRT18", "KRT8"], 
            "mesenchymal": ["COL1A1", "COL1A2", "COL6A2"],
            "immune": ["VWF","PTPRC","CD3D","CD3G","CD3E","CD79A","CD79B","CD14","FCGR3A","CD68","CD83","CSF1R","FCER1G"]}

# Load in the desired files:
for t in tissues:
    print(t)
    fpath=f"{dir}/{t}/objects/adata_PCAd_batched_umap_clustered.h5ad"
    adata = sc.read_h5ad(fpath)
    sparse = sp.sparse.csc_matrix(adata.layers['counts'])
    dense = sparse.toarray()
    X = pd.DataFrame(dense)
    X.index = adata.obs.index
    X.columns = adata.var['gene_symbols']
    adata.obs['per_gene_lineage'] = "immune"
    adata.obs['per_gene_lineage'] = adata.obs['per_gene_lineage'].astype(str)
    print("Calulating")
    sc.settings.figdir=f"{dir}/{t}/figures/UMAP"
    for g in gene_lists.keys():
        print(g)
        temp=X.iloc[:,X.columns.isin(gene_lists[g])]
        non_zero_counts = temp.apply(lambda row: (row != 0).sum(), axis=1)
        if t != "blood" and g != "mesenchymal":
            adata.obs.loc[non_zero_counts == 3, "per_gene_lineage"] = g
        adata.obs['n_' + g] = non_zero_counts.astype(str)
        adata.obs['prop_' + g] = non_zero_counts/len(gene_lists[g])   
        sc.pl.umap(adata, color = "n_" + g , save="_X_Harmony_n_" + g + "_markers.png")  
        sc.pl.umap(adata, color = "prop_" + g , save="_X_Harmony_prop_" + g + "_markers.png")  
    print("Plotting")
    sc.pl.umap(adata, color = "per_gene_lineage", save="_X_Harmony_per_gene_lineage.png")




