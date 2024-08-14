# Brad May 2024
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Load data
adata = sc.read_h5ad("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/results_round3/combined/objects/adata_PCAd_batched_umap.h5ad")
rectum = adata[adata.obs['tissue'] == "rectum"]

# Get norm prop of categories/lineages
groups = ["manual_lineage", "category__machine"]
for g in groups:
    print(g)
    lin_prop = pd.DataFrame(rectum.obs[g].value_counts(normalize=True))
    lin_prop.reset_index(inplace=True)
    print(lin_prop)
    lins = np.unique(lin_prop[g])
    lin_prop.set_index(g, inplace=True)
    lin_prop = lin_prop.T
    plt.figure(figsize=(16, 12))
    fig,ax = plt.subplots(figsize=(16,12))
    lin_prop.plot(
        kind = 'barh',
        stacked = True,
        title = 'Stacked Bar Graph',
        mark_right = True)
    plt.legend(lins, bbox_to_anchor=(1.0, 1.0))
    plt.suptitle('')
    plt.xlabel(f"Proportion of {g} in rectum")
    plt.ylabel('')
    plt.title("")
    plt.savefig(f"tissues_combined/temp/prop_{g}_rectum.png", bbox_inches='tight')
    plt.clf()

# Make QC param plots
lins = np.unique(adata.obs['manual_lineage'])
params = ["log10_n_genes_by_counts", "log10_total_counts", "pct_counts_gene_group__mito_transcript"]
for p in params:
    plt.figure(figsize=(8, 6))
    fig,ax = plt.subplots(figsize=(8,6))
    for l in lins:
        data = rectum.obs[rectum.obs.manual_lineage == l]
        data = data[p].values
        sns.distplot(data, hist=False, rug=True, label=l)
    #
    plt.legend()
    plt.xlabel(p)
    plt.savefig(f"tissues_combined/temp/rectum_{p}_per_lineage.png", bbox_inches='tight')
    plt.clf()
    

# Plot megagut over umap
figpath="tissues_combined/temp"
sc.settings.figdir=figpath
sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")
sc.pl.umap(adata, color="Celltypist:megagut_celltypist_lowerGI+lym_adult_mar24:predicted_labels", save="_megagut.png")

