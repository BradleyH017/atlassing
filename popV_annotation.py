##### Bradley September 2023
##### Annotation of the rectum using https://github.com/YosefLab/PopV
##### conda activate popV

#  Load packages
import popv
import numpy as np
import scanpy as sc
import os
import anndata as ad
import requests
import subprocess


# Changedir
import os
cwd = os.getcwd()
print(cwd)
os.chdir("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results")
cwd = os.getcwd()
print(cwd)

# Define options
data_name="rectum"
status="healthy"
category="All_not_gt_subset"
statpath = "rectum/" + status
catpath = statpath + "/" + category
figpath = catpath + "/figures"
tabdir = catpath + "/tables"
objpath = catpath + "/objects"

# Load in the QCd, PCAd rectum data (from atlassing_rectum_all_samples.py)
query_adata = ad.read_h5ad(objpath + "/adata_PCAd.h5ad")

############################
# In this instance, perform popV vs the Tabula Sapiens LI
tissue = "Large_Intestine"

query_batch_key = "experiment_id"
algorithms = None

# Initially run without known cell-type annotations
query_labels_key = None
unknown_celltype_label = "unknown"

# Load in the pre-trained data (by the authors)
res = requests.get("https://zenodo.org/api/records/7587774")
tissue_download_path = {
    ind["key"][3:-14]: ind["links"]["self"] for ind in res.json()["files"]
}
res = requests.get("https://zenodo.org/api/records/7580707")
pretrained_models_download_path = {
    ind["key"][18:-10]: ind["links"]["self"] for ind in res.json()["files"]
}
output_folder = catpath + "/PopV/tmp_testing"
refdata_url = tissue_download_path[tissue]
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

output_fn = f"{output_folder}/TS_{tissue}.h5ad"
if not os.path.exists(output_fn):
    wget_command = ["wget", "-O", output_fn, refdata_url]
    subprocess.run(wget_command, check=True, text=True)

# And now the models
model_url = pretrained_models_download_path[tissue]
output_model_tar_fn = f"{output_folder}/pretrained_model_{tissue}.tar.gz"
output_model_fn = f"{output_folder}/pretrained_model_{tissue}"
if not os.path.exists(output_model_fn):
    os.mkdir(output_model_fn)

if not os.path.exists(output_model_tar_fn):
    wget_command = ["wget", "-O", output_model_tar_fn, model_url]
    subprocess.run(wget_command, check=True, text=True)
    tar_command =  ["tar", "-xzf", output_model_tar_fn, "-C", output_model_fn]
    subprocess.run(tar_command, check=True, text=True)

# Read in the reference data
ref_adata = sc.read_h5ad(output_fn)

#Setup reference data
# Following parameters are specific to Tabula Sapiens dataset and contain the annotated cell-type and the batch_key that are corrected for during model training.
ref_labels_key = "cell_ontology_class"
ref_batch_key = "donor_assay"
min_celltype_size = np.min(ref_adata.obs.groupby(ref_labels_key).size())
n_samples_per_label = np.max((min_celltype_size, 500))

# Make gene symbols the variables in the query data
query_adata.var = query_adata.var.reset_index()
query_adata.var = query_adata.var.set_index('gene_symbols')
query_adata.var.rename(columns={'index': 'ensembl_id'}, inplace=True)

ont_dir = f"{output_folder}/ontology"
if not os.path.exists(ont_dir):
    os.mkdir(ont_dir)

# Preprocess the query with the reference dataset
# Replace X with counts
query_adata.layers['tempX'] = query_adata.X.copy()
query_adata.X = query_adata.layers['counts']
# TEMP, subsample
query_adata = sc.subsample()

from popv.preprocessing import Process_Query
adata = Process_Query(
    query_adata,
    ref_adata,
    query_labels_key=query_labels_key,
    query_batch_key=query_batch_key,
    ref_labels_key=ref_labels_key,
    ref_batch_key=ref_batch_key,
    unknown_celltype_label=unknown_celltype_label,
    save_path_trained_models=output_model_fn,
    cl_obo_folder=False,
    prediction_mode="retrain",  # 'fast' mode gives fast results (does not include BBKNN and Scanorama and makes more inaccurate predictions)
    n_samples_per_label=n_samples_per_label,
    use_gpu=0,
    compute_embedding=True,
    hvg=None,
).adata