###########################################################################################
################### Checking clusters of given resolution on UMAP #########################
###########################################################################################

import scanpy as sc
import pandas as pd 
import os
import numpy as np
import matplotlib as mp
from matplotlib import pyplot as plt
import scib_metrics 
from scib_metrics.utils import principal_component_regression

# Options
tissue="gut"
resolution="1.0"

# Set wd and load in data
os.chdir("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined")
adata = sc.read_h5ad(f"results/{tissue}/objects/adata_PCAd_batched_umap.h5ad")

# Append annotations for given resolution
adata.obs.reset_index(inplace=True)
clusters = pd.read_csv(f"results/{tissue}/tables/clustering_array/leiden_{resolution}/clusters.csv")
adata.obs = adata.obs.merge(clusters, on="cell", how="left")
adata.obs.set_index("cell", inplace=True)

# Find which clusters had poor reproducibility
keras_res = pd.read_csv(f"results/{tissue}/tables/clustering_array/leiden_{resolution}/base-model_report.tsv.gz", compression="gzip", sep = "\t")
good = keras_res.loc[keras_res['MCC'] > 0.75,'cell_label']
adata.obs['reproducible'] = adata.obs[f"leiden_{resolution}"].isin(good.values)
adata.obs[f"leiden_{resolution}"] = adata.obs[f"leiden_{resolution}"].astype(str)
adata.obs['reproducible'] = adata.obs['reproducible'].astype(str)

# Plot all clusters and annotate those not reproducible
outdir = "results/temp_xplore"
if os.path.exists(outdir) == False:
    os.mkdir(outdir)

sc.settings.figdir=outdir
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")

sc.pl.umap(adata, color = f"leiden_{resolution}", save="_" + tissue + "_leiden_" + resolution + ".png")
sc.pl.umap(adata, color = "reproducible", save="_" + tissue + "_leiden_" + resolution + "_reproducible.png")

# Plot the relationship between MCC and proportion of contributing samples
nsamps_per_cluster = adata.obs.groupby(f"leiden_{resolution}")['samp_tissue'].nunique().reset_index()
nsamps_per_cluster.columns = ['cell_label', 'unique_samp_tissue_count']
nsamps = len(np.unique(adata.obs['samp_tissue']))
nsamps_per_cluster['proportion'] = nsamps_per_cluster['unique_samp_tissue_count'] / nsamps
keras_res = keras_res.merge(nsamps_per_cluster, on="cell_label", how="left")

plt.figure(figsize=(10, 6))
plt.scatter(keras_res['proportion'], keras_res['MCC'], s=100)

# Add labels for each point
for i, label in enumerate(keras_res['cell_label']):
    plt.text(keras_res['proportion'][i], keras_res['MCC'][i], label, fontsize=8, ha='center', va='center')

# Set plot title and labels
plt.title('Scatter Plot of MCC vs Proportion')
plt.xlabel('Proportion')
plt.ylabel('MCC')
plt.savefig(f"{outdir}/MCC_proportion_contributing_samples_{tissue}.png", bbox_inches='tight')
plt.clf()


# Calculate the variance explained by the lineage and batches in the PCA
vars_check = ["samp_tissue", "sum_majority_lineage", "tissue", "category__machine"]
matrices = ["X_pca", "X_scVI"]

# Make inital df
var_df = pd.DataFrame(index=vars_check, columns=matrices)
for v in vars_check:
    for m in matrices:
        print(f"{v} - {m}")
        var_df.loc[v, m] = scib_metrics.utils.principal_component_regression(adata.obsm[m], adata.obs[v], categorical=True)



# variance explained is likely heavily influenced by the number of categories within a given variable. 
# More categories = greater likelihood of explaining variation
# So normalise these values by the number of categories
var_batch_norm = var_batch / len(np.unique(adata.obs['samp_tissue']))
var_lin_norm = var_lin / len(np.unique(adata.obs['sum_majority_lineage']))
var_lin_norm / var_batch_norm # Lineage seems to explain ~ 234 fold more *normalised* variance explained than batch does!!!