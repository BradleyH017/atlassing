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
from keras.models import Sequential
from keras.layers import Dense
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
# NN on scVI-corrected components have already been computed for this

# Downsample the adata (This machine learning approach won't be scalable with ~350k cells that we have)
# Subset to include half of the data
seed_value = 17
np.random.seed(seed_value)
all_high_qc = adata
sc.pp.subsample(adata, 0.5)

# Decide proportion to subset data with
proportion_to_subset = 2/3

###########################################################
######## Clustering of parameter optimisation data ########
###########################################################
# This is done to provide an initial set of annotations that can be used to optimise parameters against

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

# One hot encode y (the cell type classes) - like making a design matrix without intercept, encoding categorical variables as numerical
# encode class values as integers
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(y)
Y_encoded = Y_encoded.reshape(-1, 1)  # Reshape for one-hot encoding
onehot_encoder = OneHotEncoder(sparse=False)
Y_onehot = onehot_encoder.fit_transform(Y_encoded)

# Divide the 50% parameter optimisation training set into a further training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=proportion_to_subset, random_state=seed_value)


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

# 3. Perform grid search using GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_result = grid.fit(X_train, Y_train)

# Print the best parameters and their corresponding accuracy
print(f"Best: {grid_result.best_score_:.4f} using {grid_result.best_params_}")

# Make predictions and evaluate the model on the test set
Y_pred = grid_result.predict(X_test)
accuracy = accuracy_score(np.argmax(Y_test, axis=1), Y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

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


### Divide into 2/3 training 1/3 test



all_annot = grid_result.predict(X_std)















# 3. Plot the outcomes of the parameter optimisation at this resolution (This chunk is from https://github.com/andersonlab/sc_nextflow/blob/master/pipelines/0025-qc_cluster/bin/0057-scanpy_cluster_validate_resolution-keras.py)
# Make a classifier report
classes = np.unique(grid_result.predict(X_test))
y_test_pred = encoder.inverse_transform(classes)
y_test_proba = model.predict_proba(X_test)
model_report = class_report(
    y_test,
    y_test_pred,
    encoder.classes_,
    y_test_proba
)
# Add the number of cells in each class (index) in the
# (a) full dataset and (b) training dataset.
categories, counts = np.unique(y, return_counts=True)
cat_counts = dict(zip(categories, counts))
model_report['n_cells_full_dataset'] = model_report.index.map(cat_counts)
categories, counts = np.unique(y_train, return_counts=True)
cat_counts = dict(zip(categories, counts))
model_report['n_cells_training_dataset'] = model_report.index.map(
cat_counts
)

# Get a matrix of predictions on the test set
y_prob_df = pd.DataFrame(
y_test_proba,
columns=['class__{}'.format(i) for i in encoder.classes_]
)
y_prob_df['cell_label_predicted'] = y_test_pred
y_prob_df['cell_label_true'] = y_test
for i in ['cell_label_predicted', 'cell_label_true']:
    y_prob_df[i] = 'class__' + y_prob_df[i].astype(str)

score = model.evaluate(X_test, Y_test_onehot, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])