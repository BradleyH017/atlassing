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
os.chdir("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results")
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
data_name="rectum"
status="healthy"
category="All_not_gt_subset"
statpath = data_name + "/" + status
catpath = statpath + "/" + category
figpath = catpath + "/figures"
tabdir = catpath + "/tables"
objpath = catpath + "/objects"
sc.settings.figdir=figpath

# Load in the data to be annotated (cell filtered, but not gene filtered or normalised)
adata = ad.read_h5ad(objpath + "/adata_cell_filt.h5ad")
sc.pp.filter_genes(adata, min_cells=6)
# Before normalising the counts, extract the raw counts and save them
adata.raw = adata
adata.layers['counts'] = adata.X.copy()
# Calulate the CP10K expression
sc.pp.normalize_total(adata, target_sum=1e4)
# Now normalise the data to identify highly variable genes (Same as in Tobi and Monika's paper)
sc.pp.log1p(adata)


## Load the models to assign cell types (if not already obtained)
#models.download_models()
#models.models_path

# Check the model descriptions
models.models_description()

# Choose the model we want to employ
# Immune High and Immune Low has already been ran within the yascp pipeline
model = models.Model.load(model = 'Cells_Intestinal_Tract.pkl')

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

# Run the model
predictions = celltypist.annotate(adata, model = 'Cells_Intestinal_Tract.pkl', majority_voting = True)

# Transform the anndata object to include these annotations
adata = predictions.to_adata()

# Perform dotplot
celltypist.dotplot(predictions, use_as_reference = 'label', use_as_prediction = 'majority_voting', save=True)

# Save the predictions, decision matrix
predictions.probability_matrix.to_csv(tabdir + "/CellTypist_Elmentaite_prob_matrix.csv")
predictions.decision_matrix.to_csv(tabdir + "/CellTypist_Elmentaite_decision_matrix.csv")

# Generate CellTypist plots


# Plot annotations
#sc.settings.figdir=data_name + "/" + status + "/" + category + "/figures"
#sc.pl.umap(adata, color="majority_voting", frameon=True, save="_post_batch_post_sweep_celltypist_intestine_high_majority.png")

# Save object
adata.write_h5ad(objpath + "/adata_cell_filt_celltypist_intestinal.h5ad")

# Save prediction
to_save = predictions.predicted_labels
to_save.to_csv(tabdir + "/CellTypist_labels.csv")

