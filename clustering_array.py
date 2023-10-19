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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score
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
all_high_qc = adata
# NN on scVI-corrected components have already been computed for this

###########################################################
################# Clustering all data #####################
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
####### Downsample clustered data for grid search #########
###########################################################

# Downsample the adata (This grid search won't be possible with ~350k cells that we have)
# Subset to include half of the data
seed_value = 17
np.random.seed(seed_value)
sc.pp.subsample(adata, 0.5)

# Decide proportion to subset data with
proportion_to_subset = 2/3

###########################################################
############# Prep param optimisation data ################
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
cellmask = res.col.isin(adata.obs.index).values
y = y.iloc[cellmask,:]

# One hot encode y (the cell type classes) - like making a design matrix without intercept, encoding categorical variables as numerical
# encode class values as integers
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(y)
Y_encoded = Y_encoded.reshape(-1, 1)  # Reshape for one-hot encoding
onehot_encoder = OneHotEncoder(sparse=False)
Y_onehot = onehot_encoder.fit_transform(Y_encoded)

# Divide the 50% parameter optimisation training set into a further training and test set. As proportions are determined based on the size of the test set, make sure this is 1- the desired proportion of the training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=(1-proportion_to_subset), random_state=seed_value)


###########################################################
####### Perform a grid search of the paramaters ###########
###########################################################
# Have taken this code from : https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# And https://keras.io/about/
# And https://github.com/andersonlab/sc_nextflow/blob/master/pipelines/0025-qc_cluster/bin/0057-scanpy_cluster_validate_resolution-keras.py
# For the optimisation of parameters, will run this on all the dedicated data (50%) of the total number of high QC cells
# 1. Build the model (we are going to use sequential as this means using a linear stack of layers)
from keras.regularizers import l1, l2
def create_model(optimizer='adam', activation='relu', loss='categorical_crossentropy',
                 sparsity_l1_activity=0.0, sparsity_l2_activity=0.0,
                 sparsity_l1_kernel=0.0, sparsity_l2_kernel=0.0,
                 sparsity_l1_bias=0.0, sparsity_l2_bias=0.0):
    model = Sequential()
    model.add(Dense(units=Y_train.shape[1], input_dim=X_train.shape[1], activation=activation,
                    kernel_regularizer='l1_l2', bias_regularizer='l1_l2', activity_regularizer='l1_l2',
                    kernel_constraint='UnitNorm', use_bias=True))
    # Define metrics to have a look at 
    mets = [
            loss,
            keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.BinaryAccuracy(name='accuracy')
        ]
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

# Create a KerasClassifier for use with GridSearchCV
model = KerasClassifier(build_fn=create_model, verbose=0)

# 2. Define the parameters we want to assess keras accuracy accross
# Both 'softmax' and 'sigmoid' these encode output as 0 to 1 and can be interpretted as a probability
# Testing the use of the same range of sparsity as tested by leland (see https://github.com/andersonlab/sc_nextflow/blob/master/pipelines/0025-qc_cluster/bin/0057-scanpy_cluster_validate_resolution-keras.py#L704)
# sparse_categorical_crossentropy is for classes that are not one hot encoded
# Leland used the default number of epochs (100) and batch size (32) on trial of 100k TI cells
# Define the hyperparameter grid for grid search
param_grid = {
    'optimizer': ['sgd'],# 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
    'activation': ['softmax'],#, 'relu'],
    'loss': ['categorical_crossentropy'],#, 'mean_squared_error'],
    'sparsity_l1_activity': [0.0],#, 1e-6],
    'sparsity_l2_activity': [0.0],#, 1e-6],
    'sparsity_l1_kernel': [0.0],#, 1e-6],
    'sparsity_l2_kernel': [0.0],#, 1e-6],
    'sparsity_l1_bias': [0.0, 1e-6],
    'sparsity_l2_bias': [0.0, 1e-6],
    'batch_size': [32, 64],
    'epochs': [10],#, 20]
}

# Define nsplits for cross validation
n_splits = 5

# 3. Perform grid search using GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=n_splits)
grid_result = grid.fit(X_train, Y_train)

# Print the best parameters and their corresponding accuracy
print(f"Best: {grid_result.best_score_:.4f} using {grid_result.best_params_}")

# Make predictions and evaluate the model on the test set
Y_pred = grid_result.predict(X_test)
accuracy = accuracy_score(np.argmax(Y_test, axis=1), Y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Extract the desired parameters for the model at this resolution of clustering
best_params = grid_result.best_params_

###########################################################
#### Now apply this model to the rest of the data #########
###########################################################
# 1. Re-count the entire dataset
adata = all_high_qc
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

# Re-format the Y outcome
X = adata.layers['X_std']
y = res[['leiden_' + resolution]]
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(y)
Y_encoded = Y_encoded.reshape(-1, 1)  # Reshape for one-hot encoding
onehot_encoder = OneHotEncoder(sparse=False)
Y_onehot = onehot_encoder.fit_transform(Y_encoded)

# Divide into 2/3 training and 1/3 test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=(1-proportion_to_subset), random_state=seed_value)

# Use the keras best params to define the overall model
for key, value in best_params.items():
    locals()[key] = value

def create_model(optimizer=optimizer, activation=activation, epochs = epochs, loss=loss,
                 sparsity_l1_activity=sparsity_l1_activity, sparsity_l2_activity=sparsity_l2_activity,
                 sparsity_l1_kernel=sparsity_l1_kernel, sparsity_l2_kernel=sparsity_l2_kernel,
                 sparsity_l1_bias=sparsity_l1_bias, sparsity_l2_bias=sparsity_l2_bias):
    model = Sequential()
    model.add(Dense(units=Y_train.shape[1], input_dim=X_train.shape[1], activation=activation,
                    kernel_regularizer='l1_l2', bias_regularizer='l1_l2', activity_regularizer='l1_l2',
                    kernel_constraint='UnitNorm', use_bias=True))
    # Define metrics to have a look at 
    mets = [
            loss,
            keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.BinaryAccuracy(name='accuracy')
        ]
    model.compile(metrics=mets)
    return model

model = KerasClassifier(build_fn=create_model, verbose=0)
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test))

# HERE ###########

classes = model.predict(X_test)
encoder = preprocessing.LabelEncoder()
encoder.fit(y)
y_test_pred = encoder.inverse_transform(classes)
y_test_proba = model.predict_proba(X_test)



###### FROM LELAND
# Fit the specific model and save the results
model, model_report, y_prob_df, history = fit_model_keras(
            model_function=classification_model,
            encoder=encoder,
            X_std=X_std,
            y=y,
            sparsity_l1=sparsity_l1,
            sparsity_l2=0.0,
            n_epochs=n_epochs,
            batch_size=batch_size,
            train_size_fraction=train_size_fraction)


# Check accuracy on the test set



# Extract summary data and plot











