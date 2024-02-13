##################################################
#### Bradley August 2023
# CellTypist 2 to annotate the TI and rectum
# conda activate sc4
# pip install celltypist
##################################################

# Change dir
import os
cwd = os.getcwd()
print(cwd)
os.chdir("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results/tissues_combined")
cwd = os.getcwd()
print(cwd)

# Load in packages
import scanpy as sc
import celltypist
from celltypist import models
import anndata as ad
import pandas as pd
import numpy as np
import pandas as pd

# Define options and paths
tissue="blood"
use_reduction="X_pca_harmony"
use_model="/lustre/scratch127/cellgen/cellgeni/cakirb/celltypist_models/megagut_celltypist_lowerGI+lym_adult.pkl"
model_name = os.path.basename(use_model)
model_name = os.path.splitext(model_name)[0]
out_path=f"{tissue}/Annotation/CellTypist"
if os.path.exists(out_path) == False:
    os.makedirs(out_path)

sc.settings.figdir=out_path

# Load in the data to be annotated (cell filtered, batches
adata = ad.read_h5ad(fpath)

# Re-log1p CP10K the data
adata.X = adata.layers['log1p_cp10k'].copy()

## Load the models to assign cell types (if not already obtained)
#models.download_models()
#models.models_path

# Check the model descriptions
models.models_description()

# Replace the gene names with the gene symbols
if ("gene_symbols" in adata.var.columns) == False:
    import mygene 
    mg = mygene.MyGeneInfo()
    symbols = mg.querymany(adata.var.index, scopes=["ensemblgene", "symbol"])
    symbols = pd.DataFrame(symbols)
    adata.var["gene_symbols"] = symbols.symbol.values
    # Subset for those with symbol
    adata = adata[:,~pd.isnull(adata.var.gene_symbols.values)] # Losing ~ 5k genes!

adata.X.index = adata.var.gene_symbols
adata.var.index = adata.var.gene_symbols

# Copy the scVI model matrix to NN so that it is used in the model
adata.uns['distances'] = adata.uns[f"{use_reduction}_nn"]

# Run the model
predictions = celltypist.annotate(adata, model = use_model, majority_voting = True)

# Transform the anndata object to include these annotations
adata = predictions.to_adata()

# Perform dotplot
celltypist.dotplot(predictions, use_as_reference = 'label__machine', use_as_prediction = 'majority_voting', filter_prediction=0.01, save=f"dotplot_majority_voting_{model_name}_label__machine.png")
celltypist.dotplot(predictions, use_as_reference = 'label__machine', use_as_prediction = 'predicted_labels', filter_prediction=0.01, save=f"dotplot_predicted_labels_{model_name}_label__machine.png")

# Save the predictions, decision matrix
predictions.probability_matrix.to_csv(f"{out_path}/{model_name}_prob_matrix.csv")
predictions.decision_matrix.to_csv(f"{out_path}/{model_name}_decision_matrix.csv")

# Plot on UMAP
adata.obsm["X_umap"] = adata.obsm["UMAP_" + use_reduction]
sc.pl.umap(adata, color="majority_voting", frameon=True, save=f"_{model_name}_majority_voting_{use_reduction}.png")

# Save object
adata.write_h5ad(out_path + f"/adata_batched_{model_name}.h5ad")

# Save prediction
to_save = predictions.predicted_labels
to_save.to_csv(tabdir + "/CellTypist_labels.csv")

