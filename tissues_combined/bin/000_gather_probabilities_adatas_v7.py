# Brad Jan 2024: Gathering the adata's and keras probabilities of each cell being each annotation by keras
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
import gseapy as gp
from scipy import sparse
import glob
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd


tissues = ["blood", "TI", "rectum"]
dir="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/yascp_analysis/CELLRANGER_V7/annotated_anndata"
outdir="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/input_v7"

# Load in the adata for each tissue
adatas = []
for t in tissues:
    print(t)
    fpath = glob.glob(f"{dir}/*{t}*")[0]
    temp = sc.read_h5ad(fpath)
    print(temp.shape)
    temp.obs['tissue'] = t
    adatas.append(temp)

# Check vars 
all(adatas[0].var == adatas[1].var)


# Append vars (taking only the cols we want at this point)
var = adatas[0].var[['gene_symbols', 'feature_types', 'gene_group__mito_transcript',
       'gene_group__mito_protein', 'gene_group__ribo_protein',
       'gene_group__ribo_rna']]
adata = ad.concat(adatas, join="outer")
adata.var = var    
del adatas

# Add convoluted samplename for merge later
adata.obs['sanger_sample_id'] = adata.obs['Exp']

# Merge with the gut metadata sheet 
meta = pd.read_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/input_v7/gut_metadata.txt", sep="\t")
meta = meta.dropna(subset=['sanger_sample_id'])
want = ["sanger_sample_id", "sex", "age", "disease_status", "Post_QC_Genotyping_ID", "inflammation_status", "patient_id"]
adata.obs.reset_index(inplace=True)
adata.obs.rename(columns={"index": "cell"}, inplace=True)
adata.obs = adata.obs.merge(meta[want], on="sanger_sample_id", how="left")
adata.obs.set_index("cell", inplace=True)

# Add tissue to row (cell) and samp
adata.obs.reset_index(inplace=True)
adata.obs['cell'] = adata.obs['cell'] + "_" + adata.obs['tissue']
adata.obs.set_index("cell", inplace=True)
adata.obs['samp_tissue'] = adata.obs['sanger_sample_id'] + "_" + adata.obs['tissue']
# NOTE: This is currently pre-swap (CHECK THIS WITH LUCIA ONCE SHE IS DONE)

# Make sure there are not individuals aged 70 or more
adata = adata[adata.obs['age'] < 70]

# Add lineage annotation from the original keras model in the metadata
keras = adata.obs.filter(like='Keras:probability__')
keras.columns = keras.columns.str.replace('Keras:probability__', '')
keras = keras.astype(float)
keras['label__machine'] = keras.idxmax(axis=1)
keras.reset_index(inplace=True)

# Merge with category__machine annot
lab_cat= pd.read_csv("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/proc_data/data-clean_annotation.csv")
keras = keras.merge(lab_cat[["category__machine", "label__machine"]], on="label__machine", how="left")

# Derive lineage annotation:
lineages_categories = {"Immune": ['B_Cell', 'B_Cell_plasma', 'T_Cell', 'Myeloid'], 
             "Epithelial": ['Stem_cells', 'Secretory', 'Enterocyte'], 
             "Mesenchymal": "Mesenchymal"}

# Make this for lineages to labels
category_label_mapping = {}

for category, values in lineages_categories.items():
    if isinstance(values, list):
        category_label_mapping[category] = [label for value in values for label in lab_cat['label__machine'][lab_cat['category__machine'].str.contains(value)].tolist()]
    elif isinstance(values, str):
        category_label_mapping[category] = lab_cat['label__machine'][lab_cat['category__machine'].str.contains(values)].tolist()


# Convert the dictionary to a list
lineages_labels = [(name, np.unique(sublist).tolist()) for name, sublist in category_label_mapping.items()]
for lineage, labels in lineages_labels:
    print(lineage)
    temp = keras.iloc[:,keras.columns.isin(labels)]
    keras[f'sum_{lineage}'] = temp.sum(axis=1)

# Find majority lineage score
sum_columns = keras.filter(like='sum_')
keras['sum_majority_lineage'] = sum_columns.idxmax(axis=1)
keras['sum_majority_lineage'] = keras['sum_majority_lineage'].str.replace('sum_', '')

# Add to adata
adata.obs.reset_index(inplace=True)
adata.obs = adata.obs.merge(keras[["cell", "label__machine", "category__machine", "sum_Immune", "sum_Epithelial", "sum_Mesenchymal", "sum_majority_lineage"]], on="cell", how="left")
adata.obs.set_index("cell", inplace=True)

# The final shape of the data:
print(f"The final shape of the data is: {adata.shape}")

# Finally save
adata.write(f"{outdir}/adata_raw_input_all.h5ad")
