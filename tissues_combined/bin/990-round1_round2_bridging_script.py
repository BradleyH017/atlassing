#!/usr/bin/env python
####################################################################################################################################
####################################################### Bradley April 2024 #########################################################
# Bridging script to define lineage from the coarse clusters (res=0.1) defined from the first round of analysis ####################
# To be ran after first round of analysis (Snakefile with config.yaml), and before second round (Snakefile with config_in_lineage) #
####################################################################################################################################
####################################################################################################################################

# Packages
import scanpy as sc
import pandas as pd
import numpy as np

# Define tissue and resolution to define lineages from (selection of clusters for each lineage will vary depending on this)
tissue="all"
resolution="0.025"

# Import data
fpath = f"results/{tissue}/tables/clustering_array/leiden_{resolution}/adata_PCAd_batched_umap_{resolution}.h5ad"
old = sc.read_h5ad(fpath)

# Plot with labels on data for ease
sc.settings.figdir=f"results/{tissue}/figures/UMAP/annotation"
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")
sc.pl.umap(old, color="leiden", legend_loc="on data", save=f"_{tissue}_leiden_{resolution}_legend_on.png")

# Generate additional UMAPs from non HVG expression object
umap = old.obsm['UMAP_X_scVI']
clusters = old.obs['leiden']
del old
adata = sc.read_h5ad("input_v7/adata_raw_input_all.h5ad")
adata = adata[adata.obs.index.isin(clusters.index)]
adata.obsm['X_umap'] = umap
# Plot missing
missing="EPCAM,KRT8,KRT18,CDH5,COL1A1,COL1A2,COL6A2,VWF,PTPRC,CD3D,CD3G,CD3E,CD79A,CD79B,CD14,FCGR3A,CD68,CD83,CSF1R,FCER1G"
missing = missing.split(",")
exprfigpath = f"results/{tissue}/figures/UMAP/expr"
sc.settings.figdir=exprfigpath
adata.layers['counts'] = adata.X.copy()
adata.X = adata.layers['log1p_cp10k'].copy()
for c in missing:
    ens=adata.var[adata.var['gene_symbols'] == c].index[0]
    sc.pl.umap(adata, layer="log1p_cp10k", color = ens, save="_scVI_" + c + ".png")

# Also plot a dotplot
adata.obs['leiden'] = clusters
major = {"Epithelial": ["EPCAM" ,'CDH1', 'KRT19', 'EPCAM'], "Mesenchymal": ["COL1A1","COL1A2","COL6A2","VWF"], 'Immune':['PTPRC'], 'B':['CD79A', 'MS4A1', 'MS4A1', 'CD79B'], 'Plasma':['MZB1', 'JCHAIN'], 'T':['CD3D', 'CD3E', 'CD3G','CCR7','IL7R', 'TRAC'], 'Myeloid':['ITGAM', 'CD14', 'CSF1R', 'TYROBP'], 'DC':['ITGAX', 'CLEC4C','CD1C', 'FCER1A', 'CLEC10A'], 'Mac':['APOE', 'C1QA', 'CD68','AIF1'], 'Mono':['FCN1','S100A8', 'S100A9', "CD14", "FCGR3A", 'LYZ'], 'Mast':["TPSAB1", 'TPSB2', "CPA3" ], 'Platelet+RBC':['GATA1', 'TAL1', 'ITGA2B', 'ITGB3']}
sc.pl.dotplot(adata, major, layer="log1p_cp10k", gene_symbols = "gene_symbols", groupby='leiden', dendrogram=False, save="_scVI_major_markers.png")

# Summarise the top 2 annotations per cluster for a range of annotation columns
annot_cols = ["Keras:predicted_celltype", "Azimuth:predicted.celltype.l1", "Azimuth:predicted.celltype.l2", "Celltypist:megagut_celltypist_lowerGI+lym_adult_mar24:predicted_labels"]
def most_and_second_most_frequent(series):
    counts = series.value_counts()
    most_common = counts.index[0]
    most_common_pct = counts.iloc[0] / counts.sum() * 100
    
    if len(counts) > 1:
        second_most_common = counts.index[1]
        second_most_common_pct = counts.iloc[1] / counts.sum() * 100
        return f"{most_common} ({most_common_pct:.1f}%), {second_most_common} ({second_most_common_pct:.1f}%)"
    else:
        return f"{most_common} ({most_common_pct:.1f}%)"

# Group by 'leiden' and apply the custom function to each annot_col
result = adata.obs.groupby('leiden')[annot_cols].agg(lambda x: most_and_second_most_frequent(x))
result.to_csv(f"results/{tissue}/tables/clustering_array/leiden_{resolution}/top_votes_other_annotations.csv")

# Look at counts
counts = pd.DataFrame(adata.obs['leiden'].value_counts())
counts.to_csv(f"results/{tissue}/tables/clustering_array/leiden_{resolution}/cluster_counts.csv")


# Generate maps of clusters --> manual lineages
manual_mapping = {"0": "Epithelial", "1": "T", "2": "Epithelial", "3": "B", "4": "Myeloid", "5": "B", "6": "Mesenchymal", "7": "Epithelial", "8": "Mast", "9": "Platelet+RBC", "10": "chuck", "11": "chuck", "12": "chuck"}

# Add to dataframe and rename the column
adata.obs['manual_lineage'] = adata.obs['leiden'].map(manual_mapping)
adata = adata[adata.obs['manual_lineage'] != "chuck"]
adata.obs = adata.obs.rename(columns={"leiden": f"leiden_{resolution}_round1_QC"}) 
if "cluster" in adata.obs.columns:
    adata.obs.drop(columns="cluster", inplace=True)

# Also remove the previous keep/remove columns
columns_to_remove = [col for col in adata.obs.columns if 'keep' in col]
adata.obs.drop(columns=columns_to_remove, inplace=True)

# Plot the manual lineage
sc.settings.figdir=f"results/{tissue}/figures/UMAP/annotation"
sc.pl.umap(adata, color="manual_lineage", legend_loc="on data", save=f"_{tissue}_manual_lineage_legend_on.png")

# Remove the embeddings, NN, re-count the expression data
adata.obsm.clear()
adata.obsp.clear()
adata.X = adata.layers['counts']
adata.varm.clear()
var_cols_to_clear = ['n_cells', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'mean', 'std']
adata.var.drop(columns=var_cols_to_clear, inplace=True)
adata.uns.clear()

# Re-index
if "cell" in adata.obs.columns.values:
    adata.obs.set_index("cell", inplace=True)

# Print the number of samples and CD samples from each tissue
tissues = np.unique(adata.obs['tissue'])
temp = adata.obs
for t in tissues:
     tempt = temp[temp['tissue'] == t]
     print(f"For {t}, there is {len(np.unique(tempt['sanger_sample_id']))} samples. A total of {tempt.shape[0]} cells")
     cd = tempt[tempt['disease_status'] == "CD"]
     print(f"{len(np.unique(cd['sanger_sample_id']))} are CD")
     print(f"From a total of {len(np.unique(tempt['patient_id']))} individuals")
    

print(f"The final number of individuals is: {len(np.unique(adata.obs['patient_id']))}")


# Save into the results dir
adata.write_h5ad(f"results/{tissue}/objects/adata_manual_lineage_clean.h5ad")

# Dvide by manual lineage and save
lins = np.unique(adata.obs['manual_lineage'])
for l in lins:
    print(l)
    temp = adata[adata.obs['manual_lineage'] == l]
    print(temp.shape)
    temp.write_h5ad(f"input_cluster_within_lineage/adata_manual_lineage_clean_{tissue}_{l}.h5ad")