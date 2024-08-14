#################################### Bradley April 2024 ####################################
########### Summary of cluster reproducibility from a single or multiple resolution ########

# Import packages
import scanpy as sc
import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

# Define tissue
tissue="rectum"

# Chdir
os.chdir("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined") 

# Get files and load
file_paths = glob.glob(f"results/{tissue}/tables/clustering_array/leiden_*")

######## Testing
#file_paths = [item for item in file_paths if '1.0' in item]

# Load MCC
mcc = []
for f in file_paths:
    temp = pd.read_csv(f"{f}/base-model_report.tsv.gz", sep = "\t", compression="gzip")
    mcc.append(temp)

mcc = pd.concat(mcc)

# Have a look at whether it is the same cells failing at each resolution
clusters = pd.read_csv(f"{file_paths[0]}/clusters.csv")
for f in file_paths[1:]:
    print(f)
    temp = pd.read_csv(f"{f}/clusters.csv")
    clusters = clusters.merge(temp, on="cell", how="left")

# Non-reproducing clusters are number 4 and 28 at resolution of 1
# Have a look at where these cells end up in the other resolutions (primarily those below this reslution)
int_clusters = [28, 4]
int_res = "1.0"
query_clusters = clusters[clusters[f"leiden_{int_res}"].isin(int_clusters)]
leiden_cols = [col for col in query_clusters.columns if col != 'cell' and not col.startswith(f'leiden_{int_res}')]
for l in leiden_cols:
    temp = query_clusters[["cell", l, f"leiden_{int_res}"]]
    print(f"~~~~~~~~~~~~~~~~~leiden_{int_res} vs {l}~~~~~~~~~~~~~~~~~~~~")
    print(temp.groupby(f"leiden_{int_res}")[l].value_counts())


# If we define lineage at this res, which lineage do these clusters look like based on their QC metrics? 
adata = sc.read_h5ad(f"results/{tissue}/tables/clustering_array/leiden_{int_res}/adata_PCAd_batched_umap_{int_res}.h5ad")
sc.settings.figdir="temp"
sc.pl.umap(adata, color = "leiden", save=f"_clusters_res{int_res}_on.png", legend_loc='on data')

# Add manual annotation onto these
immune=["11", "17", "19", "7", "31", "25", "18", "10", "20", "9", "8", "30"]
stromal=["22","33"]
epithelial=["16","1", "0", "6", "5", "13", "14", "26", "12", "3", "15", "27", "29", "2", "32", "24", "21", "23"]
lineage_mapping = {val: 'immune' for val in immune}
lineage_mapping.update({val: 'stromal' for val in stromal})
lineage_mapping.update({val: 'epithelial' for val in epithelial})
adata.obs['manual_lineage'] = adata.obs['leiden'].map(lineage_mapping)
adata.obs['manual_lineage'].fillna(adata.obs['leiden'], inplace=True)

sc.pl.umap(adata, color = "manual_lineage", save=f"_manual_lineage_res{int_res}_on.png", legend_loc='on data')

# Look at QC metrics at this level
metrics = ["log1p_n_genes_by_counts", "log1p_total_counts", "pct_counts_gene_group__mito_transcript"]
man_lins = np.unique(adata.obs['manual_lineage'])
for m in metrics:
    for l in man_lins:
        data = adata.obs[adata.obs['manual_lineage'] == l]
        sns.distplot(data[m], hist=False, rug=True, label=l)
    plt.legend()
    plt.xlabel(m)
    plt.savefig(f"temp/{m}_per_manual_lineage.png", bbox_inches='tight')
    plt.clf()





# Load the corresponding anndata, get info on sample and tissues per cluster
sc.settings.figdir="temp"
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")
from matplotlib import pyplot as plt

for f in file_paths: 
    res = f.split('/')[-1][len('leiden_'):]
    print(f"Reading adata for res:{res}")
    adata = sc.read_h5ad(f"{f}/adata_PCAd_batched_umap_{res}.h5ad")
    # Plot umap
    sc.settings.figdir="temp"
    sc.pl.umap(adata, color = "leiden", save=f"_res{res}_clusters.png")
    # Merge with MCC and plot this too
    print("Adding mcc")
    mcc_add = mcc[mcc['resolution'].astype(str) == res]
    mcc_add['leiden'] = mcc_add['cell_label'].astype(str)
    mcc_add = mcc_add[["leiden", "MCC"]]
    adata.obs.reset_index("cell", inplace=True)
    adata.obs = adata.obs.merge(mcc_add, on="leiden", how="left")
    adata.obs.set_index("cell", inplace=True)
    sc.pl.umap(adata, color = "MCC", save=f"_res{res}_MCC.png")
    # Summarise
    print("Summarising representation")
    cluster_meta = adata.obs[["samp_tissue", "tissue", "leiden", "MCC"]]
    prop_sums = []
    for v in ["samp_tissue", "tissue"]:
        proportions = cluster_meta.groupby('leiden')[v].value_counts(normalize=True).reset_index(name=f"{v}_proportion")
        max_index = proportions.groupby('leiden')[f"{v}_proportion"].idxmax()
        prop_sum = proportions.loc[max_index]
        prop_sums.append(prop_sum)
    #
    prop_sums = prop_sums[0].merge(prop_sums[1], on="leiden")
    prop_sums.columns = ["leiden"] + ['max_' + col for col in prop_sums.columns[1:]]
    mcc_add = mcc_add.merge(prop_sums, on="leiden", how="left")
    # MCC vs sample and tissue max proportions
    # Plot scatter graph
    plt.figure(figsize=(8, 6))
    plt.scatter(mcc_add['max_samp_tissue_proportion'],mcc_add['MCC'].astype(float),s=100)
    plt.xlabel('max proportion from a single samp_tissue')
    plt.ylabel('MCC')
    for i, txt in enumerate(mcc_add['leiden']):
        plt.text(mcc_add['max_samp_tissue_proportion'][i], mcc_add['MCC'][i], str(txt), ha='center', va='bottom')
    #
    plt.savefig(f"temp/MCC_vs_max_single_sample_res{res}_{tissue}.png", bbox_inches='tight')
    plt.clf()
    #
    plt.figure(figsize=(8, 6))
    plt.scatter(mcc_add['max_tissue_proportion'],mcc_add['MCC'].astype(float),s=100)
    plt.xlabel('max proportion from a single tissue')
    plt.ylabel('MCC')
    for i, txt in enumerate(mcc_add['leiden']):
        plt.text(mcc_add['max_tissue_proportion'][i], mcc_add['MCC'][i], str(txt), ha='center', va='bottom')
    #
    plt.savefig(f"temp/MCC_vs_max_single_tissue_res{res}_{tissue}.png", bbox_inches='tight')
    plt.clf()
        
        
        
    tissue_proportions = cluster_meta.groupby('leiden')['tissue'].value_counts(normalize=True).reset_index(name='tissue_proportion')
    samp_proportions = cluster_meta.groupby('leiden')['samp_tissue'].value_counts(normalize=True).reset_index(name='sample_proportion')
    max_index = tissue_proportions.groupby('leiden')['tissue_proportion'].idxmax()
    tissue_prop_sum = tissue_proportions.loc[max_index]
    # Annotate the maximum value of each of these for each metric
    
    