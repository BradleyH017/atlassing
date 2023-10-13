##################################################
############# Bradley October 2023 ###############
###### Clustering of the rectum scRNAseq #########
##################################################
# Following atlassing_rectum_all_samples

# Load libraries
import os
cwd = os.getcwd()
print(cwd)
os.chdir("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/results")
cwd = os.getcwd()
print(cwd)

# Load packages
import sys
print(sys.path)
import numpy as np
print(np.__file__)
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib as mp
from matplotlib import pyplot as plt
from matplotlib.pyplot import rc_context
import kneed as kd
import scvi
import csv
import datetime
import seaborn as sns
from formulae import design_matrices
from limix.qc import quantile_gaussianize
import matplotlib.pyplot as plt
import math
import scipy.stats as st
import re
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
print("Loaded libraries")

# Define options
data_name="rectum"
status="healthy"
category="All_not_gt_subset"

# Load in the data we want to use here (This has been through the atlassing pre-processing script, then checked and subject to parameter sweep)
n="350"
param_sweep_path = data_name + "/" + status + "/" + category + "/objects/adata_objs_param_sweep"
nn_file=param_sweep_path + "/NN_{}_scanvi.adata".format(n)
umap_file = re.sub("_scanvi.adata", "_scanvi_umap.adata", nn_file)
adata = ad.read_h5ad(umap_file)
# NN on scVI-corrected components have already been computed for this

# Divide data into 2/3 training and 1/3 train (make sure is always the same)
seed_value = 17
np.random.seed(seed_value)
proportion_to_subset = 2/3
condition = np.random.rand(len(adata)) < proportion_to_subset
# Subset the AnnData object based on the condition
train = adata[condition, :].copy()
test = adata[~condition, :].copy()

###########################################################
##################### Clustering ##########################
###########################################################

# Perform leiden clustering across a range of resolutions
resolution = "0.5"
sc.tl.leiden(train, resolution = float(resolution), neighbors_key = "scVI_nn")

# Save a dataframe of the cells and their annotation
train.obs = train.obs.rename(columns={'leiden': 'leiden_' + resolution})
res = train.obs['leiden_' + resolution]
res = res.reset_index()
res = res.rename(columns={'index': 'cell'})

# Make outdir if not already
outdir = data_name + "/" + status + "/" + category + "/tables/leiden_sweep/"
if os.path.exists(outdir) == False:
    os.mkdir(outdir)

res.to_csv(outdir + "leiden_" + resolution + ".csv")

###########################################################
####################### Keras #############################
###########################################################

# Center and scale the data (this needs to be reperformed as this subsetting means the scaling is no longer limited to just these samples)
for set in [train, test]:
    # Re-count the data
    set.X = set.layers['counts'].copy()
    sc.pp.normalize_total(set, target_sum=1e4)
    sc.pp.log1p(set)
    X = set.X.todense()
    scaler = preprocessing.StandardScaler(
        with_mean=True,
        with_std=True
    )
    X_std = scaler.fit_transform(np.asarray(X))
    set.layers['X_std'] = X_std

# Now build model to predict the expression of the remaining cells from the 
X = train.layers['X_std']
y = res[['leiden_' + resolution]]

# One hot encode y (the cell type classes) - like making a design matrix without intercept, encoding categorical variables as numerical
# encode class values as integers
encoder = preprocessing.LabelEncoder()
encoder.fit(y)
print('Found {} clusters'.format(len(encoder.classes_)))




