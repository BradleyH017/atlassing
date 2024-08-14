#!/usr/bin/env python
# -*- coding: utf-8 -*-


__date__ = '2020-04-24'
__version__ = '0.0.1'
# Script has been taken from yascp (https://github.com/wtsi-hgi/yascp/blob/1d465f848a700b9d39c4c9a63356bd7b985f5a1c/bin/0057-scanpy_cluster_validate_resolution-keras.py#L4)


print("~~~~~~~~~~~~ Running 005-scanpy_cluster_validate_resolution-keras ~~~~~~~~~~~~~~~~~")
import argparse
import os
os.environ['NUMBA_CACHE_DIR']='/tmp'
os.environ['MPLCONFIGDIR']='/tmp'
import random
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
import csv
from distutils.version import LooseVersion

# import joblib  # for numpy matrix, joblib faster than pickle
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import plotnine as plt9

from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2
from keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.python.client import device_lib
import tensorflow as tf

print("Loaded modules")

# Create custom h5 read in functions? (https://github.com/broadinstitute/CellBender/issues/128#issuecomment-1175336065):

# Check that we are working on GPU or CPU
# print(device_lib.list_local_devices())  # list of DeviceAttributes
# tf.config.list_physical_devices('GPU')

# Set seed for reproducibility
seed_value = 0
# 0. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 1. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 2. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 3. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)

# Get compression opts for pandas
compression_opts = 'gzip'
if LooseVersion(pd.__version__) > '1.0.0':
    compression_opts = dict(method='gzip', compresslevel=9)


def _create_colors(classes):
    n_cts = len(classes)
    color_norm = colors.Normalize(vmin=-n_cts / 3, vmax=n_cts)
    ct_arr = np.arange(n_cts)
    ct_colors = cm.YlGnBu(color_norm(ct_arr))
    #
    return ct_colors


def plot_roc(y_prob, y_test, classes):
    """Plot ROC curve. Based off of NaiveDE library."""
    ct_colors = _create_colors(classes)
    #
    for i, cell_type in enumerate(classes):
        fpr, tpr, _ = metrics.roc_curve(y_test == cell_type, y_prob[:, i])
        plt.plot(fpr, tpr, c=ct_colors[i], lw=2)
    #
    plt.plot([0, 1], [0, 1], color='k', ls=':')
    plt.xlabel('FPR')
    plt.ylabel('TPR')


def class_report(y_true, y_pred, classes, y_pred_proba=None):
    """
    Build a text report showing the main classification metrics.
    
    Replaces sklearn.metrics.classification_report.

    Derived from:
    https://stackoverflow.com/questions/39685740/calculate-sklearn-roc-auc-score-for-multi-class
    """
    if y_true.shape != y_pred.shape:
        raise Exception(
            'Error! y_true {} is not the same shape as y_pred {}'.format(
                y_true.shape,
                y_pred.shape
            )
        )
    #
    # NOTE: Y may not have predictions for all classes
    model_report = pd.DataFrame(classification_report(
        y_true,
        y_pred,
        classes,
        output_dict=True
    )).transpose()
    #
    if not (y_pred_proba is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        aupc = dict()
        mcc = dict()
        for label_it, label in enumerate(model_report.index):
            if label in classes.astype(str):  # skip accuracy, macro avg, weighted avg
                fpr[label], tpr[label], _ = metrics.roc_curve(
                    (y_true == label).astype(int),
                    y_pred_proba[:, label_it]
                )
                roc_auc[label] = metrics.auc(fpr[label], tpr[label])
                aupc[label] = metrics.average_precision_score(
                    (y_true == label).astype(int),
                    y_pred_proba[:, label_it],
                    average=None  # No need since iter over labels
                )
                mcc[label] = metrics.matthews_corrcoef(
                    (y_true == label).astype(int),
                    (y_pred == label).astype(int)
                )
            else:
                fpr[label] = np.nan
                tpr[label] = np.nan
                roc_auc[label] = np.nan
                aupc[label] = np.nan
                mcc[label] = np.nan
        #
        model_report['AUC'] = pd.Series(roc_auc)
        model_report['average_precision_score'] = pd.Series(aupc)
        model_report['MCC'] = pd.Series(mcc)
        #
        # Catch the case where true label not predicted in lr, perhaps because
        # too few training cases.
        for i in np.unique(y_true):
            if i not in model_report.index:
                print(
                    'Adding category ({}) from {}.'.format(
                        i,
                        'truth with no prediction to report'
                    )
                )
                model_report = model_report.append(pd.Series(
                    [np.nan]*len(model_report.columns),
                    index=model_report.columns,
                    name=i
                ))
        #
        #model_report = model_report.sort_index()
        #
        model_report = model_report.dropna(how='all')
        #
        return model_report

def keras_grid(
    model_function,
    encoder,
    X_std,
    y,
    n_epochs=100,
    batch_size=32
):
    # Run same proceedure on the test data
    y_encoded = encoder.transform(y)
    Y_onehot = np_utils.to_categorical(y_encoded)
    # Initial parameter sweep on 100k cells
    # Explanation of params chosen for optimisation:
    # activation: 'softmax' - Each nput vector is handled independently, the vector of values is converted into a probablity from 0-1 (exactly what we want). The result can be interpretted as a probability distribution (https://keras.io/api/layers/activations/)
    # loss: 'categorical_crossentropy' - As our data is not binary, much be one of category_crossentropy or sparse_categorical_crossentropy (https://keras.io/2.15/api/). 
    # # sparse_categorical_crossentropy is for classes that are not one-hot-encoded (e.g, can be part of multiple classes), so not appropraite.
    # optimizer: ['adam', 'sgd', 'RMSprop'] - Adam supposed to generally work well for most problems. sgd used in original TI model and is a simple model, rmsprop is suitable for problems with sparse data or non-stationary objectives
    # sparsity_l1/2_activity: Using same params as tried originally
    # sparsity_l1/2_kernal: Using same params as tried originally
    # sparsity_l1/2_bias: Using same params as tried originally
    param_grid = dict(
        activation=['softmax'],
        optimizer=['sgd'],
        loss=['categorical_crossentropy'],
        sparsity_l2__activity=[0.0, 0.01, 1e-4],
        sparsity_l1__activity=[0.0, 0.01, 1e-4],
        sparsity_l2__kernel=[0.0, 0.01, 1e-4],
        sparsity_l1__kernel=[0.0, 0.01, 1e-4],
        sparsity_l2__bias=[0.0, 0.01, 1e-4],
        sparsity_l1__bias=[0.0, 0.01, 1e-4]
    )
    n_splits = 5

    ########### TESTING ############
    
    grid = GridSearchCV(
        estimator=KerasClassifier(build_fn=model_function),
        param_grid=param_grid,
        n_jobs=-2, # Leave all but one available core for parallelisation
        cv=n_splits  # Number of cross validation.
    )
    # NOTE: We could pass batch_size and epochs here, but we get results much
    # faster if we just use the defaults.
    grid_result = grid.fit(
        # batch_size=batch_size,
        # epochs=n_epochs,
        X=X_std,
        y=Y_onehot
    )
    #
    # Make a dataframe of the results of all of the models.
    cv_results = grid_result.cv_results_.copy()
    del cv_results['param_activation']
    df_grid_result = pd.DataFrame(cv_results.pop('params'))
    # Rename so we know that these columns are parameters
    df_grid_result.columns = [
        'param__{}'.format(i) for i in df_grid_result.columns
    ]
    df_grid_result = pd.concat([
        df_grid_result,
        pd.DataFrame(cv_results)
    ], axis=1)
    #
    print('Best: %f using %s' % (
        grid_result.best_score_,
        grid_result.best_params_
    ))
    #
    return grid_result, df_grid_result



def main():
    """Run CLI."""
    parser = argparse.ArgumentParser(
        description="""
            Fits logistic regression to predict labels.'
            """
    )
    
    parser.add_argument(
        '-t', '--tissue',
        action='store',
        dest='tissue',
        required=True,
        help=''
    )

    parser.add_argument(
        '-sp', '--sparse_matrix_path',
        action='store',
        dest='sparse_matrix_path',
        required=True,
        help='sparse matrix to run keras on'
    )
    
    parser.add_argument(
        '-cl', '--clusters_df',
        action='store',
        dest='clusters_df',
        required=True,
        help='Matrix with cells,clusters columns'
    )
    
    parser.add_argument(
        '-g', '--genes_f',
        action='store',
        dest='genes_f',
        required=True,
        help='genes file'
    )
    
    parser.add_argument(
        '-cf', '--cells_f',
        action='store',
        dest='cells_f',
        required=True,
        help='cell file'
    )

    parser.add_argument(
        '-nepoch', '--number_epoch',
        action='store',
        dest='number_epoch',
        default=25,
        type=int,
        help='Number of epochs.\
            (default: %(default)s)'
    )

    parser.add_argument(
        '-bs', '--batch_size',
        action='store',
        dest='batch_size',
        default=32,
        type=int,
        help='Batch size. Divides the dataset into n batches and updates the\
            weights at the end of each ne.\
            (default: %(default)s)'
    )
    
    parser.add_argument(
        '-tsf', '--train_size_fraction',
        action='store',
        dest='train_size_fraction',
        default=0.67,
        type=float,
        help='Fraction of the data to use for training set.\
            (default: %(default)s)'
    )

    parser.add_argument(
        '-of', '--output_file',
        action='store',
        dest='of',
        default='',
        help='Basename of output files, assuming output in current working \
            directory.\
            (default: keras_model-<params>)'
    )
    options = parser.parse_args()
    tissue = options.tissue
    print(f"tissue: {tissue}")
    sparse_matrix_path = options.sparse_matrix_path
    print(f"sparse_matrix_path: {sparse_matrix_path}")
    clusters_df = options.clusters_df
    print(f"clusters_df: {clusters_df}")
    
    # Testing
    # tissue="blood"
    # sparse_matrix_path=f"/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Atlassing/tissues_combined/results/{tissue}/tables/log1p_cp10k_sparse.npz"
    # clusters_df = f"results/{tissue}/tables/clustering_array/leiden_3.0/clusters.csv"
    # genes_f=f"results/{tissue}/tables/genes.txt"
    # cells_f=f"results/{tissue}/tables/cells.txt"
    # out_f=f"results/{tissue}/tables/keras-grid_search/keras-"


    verbose = True

    # Set GPU memory limits
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        # For TF v1
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # session = tf.Session(config=config)

        # For TF v2
        try:
            # Method 1:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Method 2:
            # Restrict TensorFlow to only allocate 1GB of memory on the first
            # GPU
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpus[0],
            #     [tf.config.experimental.VirtualDeviceConfiguration(
            #         memory_limit=options.memory_limit*1024
            #     )])
            # logical_gpus = tf.config.list_logical_devices('GPU')
            # print(
            #     len(gpus),
            #     "Physical GPUs,",
            #     len(logical_gpus),
            #     "Logical GPUs"
            # )
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    else:
        _ = 'running without gpus'
        # raise Exception('ERROR: no GPUs detected.')

    # Load the expression and make dense
    sparse = sp.sparse.load_npz(sparse_matrix_path)
    dense = sparse.toarray()
    # Create a pandas DataFrame from the dense matrix
    X = pd.DataFrame(dense)
    del dense

    # Add genes to the columns and cells to the rows
    genes = np.loadtxt(options.genes_f, dtype=str)
    cells = np.loadtxt(options.cells_f, dtype=str)
    X.columns = genes
    X.index = cells

    # Load the clusters and add to 'y'
    clusters = pd.read_csv(clusters_df, dtype=object)
    clusters.set_index("cell",inplace=True)
    leiden_column = [col for col in clusters.columns if col.startswith('leiden')][0]
    clusters[leiden_column] = clusters[leiden_column].astype(str)
    clusters[leiden_column] = clusters[leiden_column].astype('category')
    y = clusters[leiden_column].values
        
    # If the dataset is greater than 100k cells, subset to only include these cells in the grid search
    max_cells=100000
    if len(cells) > max_cells:
        # Generate a vector of random indices for sampling
        random_indices = pd.Series(range(len(cells))).sample(n=max_cells, replace=False)
        X = X.iloc[random_indices]
        y = y[random_indices]

    # Set other variables
    n_epochs = options.number_epoch
    batch_size = options.batch_size

    # Center and scale the data
    if sp.sparse.issparse(X):
        X = X.todense()

    X_std = X
    scaler = preprocessing.StandardScaler(
        with_mean=True,
        with_std=True
    )

    X_std = scaler.fit_transform(X)
    if verbose:
        print('center={} scale={}'.format(
            True,
            True
        ))

    # One hot encode y (the cell type classes)
    # encode class values as integers
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y)
    print('Found {} clusters'.format(len(encoder.classes_)))

    # Define the model
    # NOTE: Defaults determined via grid search of 160k TI single cells
    def classification_model(
        optimizer='sgd',
        activation='softmax',
        loss='categorical_crossentropy',
        sparsity_l1__activity=0.0001,
        sparsity_l2__activity=0.0,
        sparsity_l1__kernel=0.0,
        sparsity_l2__kernel=0.0,
        sparsity_l1__bias=0.0,
        sparsity_l2__bias=0.0
    ):
        # create model
        model = Sequential()
        # Use a “softmax” activation function in the output layer. This is to
        # ensure the output values are in the range of 0 and 1 and may be used
        # as predicted probabilities.
        #
        # https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax
        # Softmax assigns decimal probabilities to each class in a multi-class
        # problem. Those decimal probabilities must add up to 1.0. This
        # additional constraint helps training converge more quickly than it
        # otherwise would. Softmax is implemented through a neural network
        # layer just before the output layer. The Softmax layer must have the
        # same number of nodes as the output layer.
        # Softmax assumes that each example is a member of exactly one class.
        #
        # Softmax should be used for multi-class prediction with single label
        # https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/video-lecture
        # NOTE: input dimension = number of features your data has
        model.add(Dense(
            len(encoder.classes_),  # output dim is number of classes
            use_bias=True,  # intercept
            activation=activation,  # softmax, sigmoid
            activity_regularizer=L1L2(
                l1=sparsity_l1__activity,
                l2=sparsity_l2__activity
            ),
            kernel_regularizer=L1L2(
                l1=sparsity_l1__kernel,
                l2=sparsity_l2__kernel
            ),
            bias_regularizer=L1L2(
                l1=sparsity_l1__bias,
                l2=sparsity_l2__bias
            ),
            input_dim=X.shape[1]
        ))
        # Example of adding additional layers
        # model.add(Dense(8, input_dim=4, activation='relu'))
        # model.add(Dense(3, activation='softmax'))
        #
        # Metrics to check out over training epochs
        mets = [
            # loss,
            keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
            # keras.metrics.TruePositives(name='tp'),
            # keras.metrics.FalsePositives(name='fp'),
            # keras.metrics.TrueNegatives(name='tn'),
            # keras.metrics.FalseNegatives(name='fn'),
            # keras.metrics.Precision(name='precision'),
            # keras.metrics.Recall(name='recall'),
            # keras.metrics.AUC(name='auc'),
            keras.metrics.BinaryAccuracy(name='accuracy')
        ]
        # Use Adam gradient descent optimization algorithm with a logarithmic
        # loss function, which is called “categorical_crossentropy” in Keras.
        # UPDATE: sgd works better emperically.
        model.compile(
            optimizer=optimizer,  # adam, sgd
            loss=loss,
            metrics=mets
        )
        #
        return model
        
    # Get the out file base.
    out_file_base = options.of

    #
    # Call grid search of various parameters
    grid_result, df_grid_result = keras_grid(
        model_function=classification_model,
        encoder=encoder,
        X_std=X_std,
        y=y,
        n_epochs=n_epochs,
        batch_size=batch_size
    )

    out_f = '{}-grid_result.tsv.gz'.format(out_file_base)
    df_grid_result.to_csv(
        out_f,
        sep='\t',
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
        na_rep='',
        compression=compression_opts
    )
    #
    # Add a single columns that summarizes params
    param_columns = [
        col for col in df_grid_result.columns if 'param__' in col
    ]
    df_grid_result['params'] = df_grid_result[
        param_columns
    ].astype(str).apply(lambda x: '-'.join(x), axis=1)
    #
    # Plot the distribution of accuracy across folds
    split_columns = [
        col for col in df_grid_result.columns if 'split' in col
    ]
    split_columns = [
        col for col in split_columns if '_test_score' in col
    ]
    df_plt = pd.melt(
        df_grid_result,
        id_vars=['params'],
        value_vars=split_columns
    )
    gplt = plt9.ggplot(df_plt, plt9.aes(
        x='params',
        y='value'
    ))
    gplt = gplt + plt9.theme_bw()
    gplt = gplt + plt9.geom_boxplot(alpha=0.8)
    gplt = gplt + plt9.geom_jitter(alpha=0.75)
    gplt = gplt + plt9.scale_y_continuous(
        # trans='log10',
        # labels=comma_labels,
        minor_breaks=0
        # limits=[0, 1]
    )
    gplt = gplt + plt9.labs(
        x='Parameters',
        y='Score',
        title=''
    )
    gplt = gplt + plt9.theme(
        axis_text_x=plt9.element_text(angle=-45, hjust=0)
    )
    gplt.save(
        '{}-score.png'.format(out_file_base),
        #dpi=300,
        width=10,
        height=4,
        limitsize=False
    )
    #
    # Plot the mean time and std err for fitting results
    gplt = plt9.ggplot(df_grid_result, plt9.aes(
        x='params',
        y='mean_fit_time'
    ))
    gplt = gplt + plt9.theme_bw()
    gplt = gplt + plt9.geom_point()
    gplt = gplt + plt9.geom_errorbar(
        plt9.aes(
            ymin='mean_fit_time-std_fit_time',
            ymax='mean_fit_time+std_fit_time'
        ),
        width=0.2,
        position=plt9.position_dodge(0.05)
    )
    gplt = gplt + plt9.scale_y_continuous(
        # trans='log10',
        # labels=comma_labels,
        minor_breaks=0
    )
    gplt = gplt + plt9.labs(
        x='Parameters',
        y='Mean fit time',
        title=''
    )
    gplt = gplt + plt9.theme(
        axis_text_x=plt9.element_text(angle=-45, hjust=0)
    )
    gplt.save(
        '{}-fit_time.png'.format(out_file_base),
        #dpi=300,
        width=10,
        height=4,
        limitsize=False
    )

    # Save a file that indicates what the params to take forward are. This should be derived from the test with the best score
    best_params = df_grid_result[df_grid_result['rank_test_score'] == 1]
    best_params = best_params.iloc[0,:]
    pd.DataFrame(best_params[0:9]).to_csv(f"{out_file_base}use_params.txt", sep = "\t", header=None)


if __name__ == '__main__':
    main()