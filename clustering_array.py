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
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.regularizers import L1L2
from keras.wrappers.scikit_learn import KerasClassifier
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

# Downsample the adata (This machine learning approach won't be scalable with ~350k cells that we have)
# Subset to include half of the data
seed_value = 17
np.random.seed(seed_value)
sc.pp.subsample(adata, 0.5)

# Decide proportion to subset data with
proportion_to_subset = 2/3

###########################################################
##################### Clustering ##########################
###########################################################

# Perform leiden clustering across a range of resolutions
resolution = "0.5"
sc.tl.leiden(adata, resolution = float(resolution), neighbors_key = "scVI_nn")

# Save a dataframe of the cells and their annotation
adata.obs = adata.obs.rename(columns={'leiden': 'leiden_' + resolution})
res = adata.obs['leiden_' + resolution]
res = res.reset_index()
res = res.rename(columns={'index': 'cell'})

# Make outdir if not already
outdir = data_name + "/" + status + "/" + category + "/tables/leiden_sweep/"
if os.path.exists(outdir) == False:
    os.mkdir(outdir)

res.to_csv(outdir + "leiden_" + resolution + ".csv")

###########################################################
####################### Prep data #########################
###########################################################

# Re-count the data
adata.X = adata.layers['counts'].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
X = adata.X.todense()
scaler = preprocessing.StandardScaler(
    with_mean=True,
        with_std=True
)
X_std = scaler.fit_transform(np.asarray(X))
adata.layers['X_std'] = X_std

# Now build model to predict the expression of the remaining cells from the 
X = adata.layers['X_std']
y = res[['leiden_' + resolution]]

# One hot encode y (the cell type classes) - like making a design matrix without intercept, encoding categorical variables as numerical
# encode class values as integers
encoder = preprocessing.LabelEncoder()
encoder.fit(y)
Y=encoder.transform(y)
print('Found {} clusters'.format(len(encoder.classes_)))

###########################################################
#### Perform a grid search of the paramaters for keras ####
###########################################################
# Have taken this code from : https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# And https://keras.io/about/
# 1. Build the model (we are going to use sequential as this means using a linear stack of layers)
def create_model():
    # Initialise sequential
    model = Sequential()
    # Add layers
    model.add(Dense(units=len(encoder.classes_), 
                use_bias=True,  # intercept 
                input_shape=(X_std.shape[0],), 
                activation='softmax'
    ))
    # Compile
    model.compile(metrics=['accuracy'])
    # return
    return model

# 2. Define the parameters we want to assess keras accuracy accross
# Both 'softmax' and 'sigmoid' these encode output as 0 to 1 and can be interpretted as a probability
# Testing the use of the same range of sparsity as tested by leland (see https://github.com/andersonlab/sc_nextflow/blob/master/pipelines/0025-qc_cluster/bin/0057-scanpy_cluster_validate_resolution-keras.py#L704)
# sparse_categorical_crossentropy is for classes that are not one hot encoded
# Leland used the default number of epochs (100) and batch size (32) on trial of 100k TI cells
param_grid = dict(
    epochs=[10, 50, 100], 
    batch=[10, 20, 32, 40, 60, 80, 100],
    activation=['softmax', 'relu'],
    optimizer=['sgd','SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
    loss=['categorical_crossentropy', 'mean_squared_error'],
    sparsity_l2__activity=[0.0, 1e-6],
    sparsity_l1__activity=[0.1, 1e-4, 1e-10, 0.0],
    sparsity_l2__kernel=[0.0, 1e-6],
    sparsity_l1__kernel=[0.1, 1e-4, 1e-10, 0.0],
    sparsity_l2__bias=[0.0, 1e-6],
    sparsity_l1__bias=[0.1, 1e-4, 1e-10, 0.0]
)

param_grid = dict(
    optimizer=['sgd','SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
    loss=['categorical_crossentropy', 'mean_squared_error'],
)


# 3. Define the number of cross validations to perform
n_splits = 5

# 4. Keras the model
model = KerasClassifier(build_fn=create_model, 
        verbose=0)

# 5.  
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    n_jobs=-1,
    cv=n_splits  # Number of cross validation.
)

# 7. Fit the model
grid_result = grid.fit(X, Y)

