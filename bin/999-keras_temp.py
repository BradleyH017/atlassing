
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
    #
    # Initial parameter sweep for different activation, optimizer, and loss.
    # NOTE: From 100k TI single cells, best settings were:
    # 'activation': 'softmax',
    # 'loss': 'categorical_crossentropy',
    # 'optimizer': toss up between adam and sgd, though sgd generally better
    # 'sparsity_l1': 0.001
    # param_grid = dict(
    #     activation=['softmax', 'sigmoid'],
    #     optimizer=['sgd', 'adam'],
    #     loss=['categorical_crossentropy', 'mean_squared_error'],
    #     sparsity_l1=[0.1, 0.01, 0.001, 0.0005]
    # )
    # NOTE: sparse_categorical_crossentropy is for classes that are not one
    # hot encoded.
    # https://www.quora.com/What-is-the-difference-between-categorical_crossentropy-and-sparse_categorical-cross-entropy-when-we-do-multiclass-classification-using-convolution-neural-networks
    # param_grid = dict(
    #     activation=['softmax'],
    #     optimizer=['sgd'],
    #     loss=['categorical_crossentropy'],
    #     sparsity_l2__activity=[0.0, 1e-6],
    #     sparsity_l1__activity=[0.1, 1e-4, 1e-10, 0.0],
    #     sparsity_l2__kernel=[0.0, 1e-6],
    #     sparsity_l1__kernel=[0.1, 1e-4, 1e-10, 0.0],
    #     sparsity_l2__bias=[0.0, 1e-6],
    #     sparsity_l1__bias=[0.1, 1e-4, 1e-10, 0.0]
    # )
    param_grid = dict(
        activation=['softmax'],
        optimizer=['sgd'],
        loss=['categorical_crossentropy'],
        sparsity_l2__activity=[0.0],
        sparsity_l1__activity=[0.1, 1e-4],
        sparsity_l2__kernel=[0.0],
        sparsity_l1__kernel=[0.1, 1e-4],
        sparsity_l2__bias=[0.0],
        sparsity_l1__bias=[0.1, 1e-4]
    )
    n_splits = 5
    grid = GridSearchCV(
        estimator=KerasClassifier(build_fn=model_function),
        param_grid=param_grid,
        n_jobs=1,
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


def fit_model_keras(
    model_function,
    encoder,
    X_std,
    y,
    keras_params,
    n_epochs=100,
    batch_size=32,
    train_size_fraction=0.67,
    verbose=True
):
    # References:
    # https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
    # https://stackoverflow.com/questions/59643062/scikit-learn-vs-keras-tensorflow-for-multinomial-logistic-regression
    # https://medium.com/@luwei.io/logistic-regression-with-keras-d75d640d175e
    #
    # Make the training and test dataset
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_std,
        y,
        stratify=y,
        random_state=61,
        train_size=train_size_fraction
    )
    if verbose:
        print(
            'Split X into training {} and test {} sets.'.format(
                X_train.shape,
                X_test.shape
            )
        )
    #
    # One hot encode y (the cell type classes)
    # encode class values as integers
    # encoder = preprocessing.LabelEncoder()
    # encoder.fit(y_train)
    y_train_encoded = encoder.transform(np.ravel(y_train))
    # convert integers to dummy variables (i.e. one hot encoded)
    Y_train_onehot = np_utils.to_categorical(y_train_encoded)
    # Run same proceedure on the test data
    y_test_encoded = encoder.transform(np.ravel(y_test))
    Y_test_onehot = np_utils.to_categorical(y_test_encoded)
    #
    # Training
    model = model_function(
        optimizer=keras_params['optimizer'],
        activation=keras_params['activation'],
        loss=keras_params['loss'],
        sparsity_l1__activity=keras_params['sparsity_l1__activity'],
        sparsity_l2__activity=keras_params['sparsity_l2__activity'],
        sparsity_l1__kernel=keras_params['sparsity_l1__kernel'],
        sparsity_l2__kernel=keras_params['sparsity_l2__kernel'],
        sparsity_l1__bias=keras_params['sparsity_l1__bias'],
        sparsity_l2__bias=keras_params['sparsity_l2__bias']
    )
    history = model.fit(
        X_train,
        Y_train_onehot,
        batch_size=batch_size,
        epochs=n_epochs,
        verbose=0,
        # use_multiprocessing=True,
        # validation_split=0.33  # Frac of the training used for validation.
        validation_data=(X_test, Y_test_onehot)
    )
    #
    # Train using KFold validation
    # from keras.wrappers.scikit_learn import KerasClassifier
    # from sklearn.model_selection import KFold
    # from sklearn.model_selection import cross_val_score
    # estimator = KerasClassifier(
    #     build_fn=classification_model,
    #     epochs=200,
    #     # batch_size=5,
    #     verbose=1
    # )
    # kfold = KFold(n_splits=10, shuffle=True)
    # results = cross_val_score(estimator, X_std, y_onehot, cv=kfold)
    # print("Baseline: %.2f%% (%.2f%%)" % (
    #     results.mean()*100, results.std()*100)
    # )
    #
    # Make a classifier report
    classes = np.argmax(model.predict(X_test), axis=1)
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
    categories, counts = np.unique(y.astype(str), return_counts=True)
    cat_counts = dict(zip(categories, counts))
    model_report['n_cells_full_dataset'] = model_report.index.map(cat_counts)
    categories, counts = np.unique(y_train.astype(str), return_counts=True)
    cat_counts = dict(zip(categories, counts))
    model_report['n_cells_training_dataset'] = model_report.index.map(
        cat_counts
    )
    #
    # Get a matrix of predictions on the test set
    y_prob_df = pd.DataFrame(
        y_test_proba,
        columns=['class__{}'.format(i) for i in encoder.classes_]
    )
    y_prob_df['cell_label_predicted'] = y_test_pred
    y_prob_df['cell_label_true'] = y_test
    for i in ['cell_label_predicted', 'cell_label_true']:
        y_prob_df[i] = 'class__' + y_prob_df[i].astype(str)
    #
    #score = model.evaluate(X_test, Y_test_onehot, verbose=0)
    #print('Test score:', score[0])
    #print('Test accuracy:', score[1])
    #
    return model, model_report, y_prob_df, history


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
        '-v', '--version',
        action='version',
        version='%(prog)s {version}'.format(version=__version__)
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
        '-kpf', '--keras_param_file',
        action='store',
        dest='keras_param_file',
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
            weights at the end of each one.\
            (default: %(default)s)'
    )

    parser.add_argument(
        '-tsc', '--train_size_cells',
        action='store',
        dest='train_size_cells',
        default=0,
        type=int,
        help='Number of cells to use for training set. If > 0 all\
            remaining cells not randomly selected for training will be used\
            for the test set. Overrides <train_size_fraction>.\
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
        '--dict_add',
        action='store',
        dest='dict_add',
        default='',
        type=str,
        help='Additional information to add to output model_report.\
            Format: key::value:::key2::value2.\
            Example: method::leiden:::resolution::3.0\
            (default: %(default)s)'
    )

    parser.add_argument(
        '--grid_search',
        action='store_true',
        dest='grid_search',
        default=False,
        help='Run a grid search of hyperparameters.\
            (default: %(default)s)'
    )

    parser.add_argument(
        '--memory_limit',
        action='store',
        dest='memory_limit',
        default=50,
        type=int,
        help='Memory limit in Gb.\
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
    keras_param_file = options.keras_param_file
    print(f"keras_param_file: {keras_param_file}")
    print("Done")
    
    



### Testing
tissue = "lucia"
sparse_matrix_path = "lucia_temp/log1p_cp10k_sparse.npz"
clusters_df = "lucia_temp/clusters.csv"
genes_f="lucia_temp/raw_genes.txt"
cells_f="lucia_temp/raw_cells.txt"
keras_param_file = "lucia_temp/keras-use_params.txt"
n_epochs=25 
batch_size=32
train_size_fraction=0.67
out_file_base="lucia_temp/base_0804"

# Load in keras params
keras_params_df = pd.read_csv(keras_param_file, sep = "\t", header=None)
keras_params_df.columns = ["param", "value"]
keras_params_df['param'] = keras_params_df['param'].str.replace('param__', '')
keras_params = {}
# Then, iterate over the DataFrame rows
for index, row in keras_params_df.iterrows():
    # Get the values from the row
    param = row['param']
    value = row['value']
    # Check if the parameter is 'activation' or 'loss'
    if param == 'activation' or param == 'loss' or param == 'optimizer':
        # If it is, add it as a string variable
        keras_params[param] = str(value)
    else:
        # Otherwise, add it as a float variable
        keras_params[param] = float(value)

print(f"keras_params = {keras_params}")


verbose = True


# Load in expression, genes, cells, clusters
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
clusters = pd.read_csv(clusters_df)
clusters.set_index('cell', inplace=True)
leiden_column = [col for col in clusters.columns if col.startswith('leiden')][0]
clusters[leiden_column] = pd.Categorical(clusters[leiden_column], categories=range(max(clusters[leiden_column])+1), ordered=False)
clusters[leiden_column] =  'c' + clusters[leiden_column].astype(str)
y = clusters[leiden_column].values
print(y)
    
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
    
def classification_model(
        optimizer=keras_params['optimizer'],
        activation=keras_params['activation'],
        loss=keras_params['loss'],
        sparsity_l1__activity=keras_params['sparsity_l1__activity'],
        sparsity_l2__activity=keras_params['sparsity_l2__activity'],
        sparsity_l1__kernel=keras_params['sparsity_l1__kernel'],
        sparsity_l2__kernel=keras_params['sparsity_l2__kernel'],
        sparsity_l1__bias=keras_params['sparsity_l1__bias'],
        sparsity_l2__bias=keras_params['sparsity_l2__bias']
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
            optimizer=keras_params['optimizer'],  # adam, sgd
            loss=keras_params['loss'],
            metrics=mets
        )
        #
        return model

X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_std,
        y,
        stratify=y,
        random_state=61,
        train_size=train_size_fraction
    )

y_train_encoded = encoder.transform(y_train) # potential alternative (y_train.codes) - (encoder.transform(y_train))
# convert integers to dummy variables (i.e. one hot encoded)
Y_train_onehot = np_utils.to_categorical(y_train_encoded)
# Run same proceedure on the test data
y_test_encoded = encoder.transform(y_test)
Y_test_onehot = np_utils.to_categorical(y_test_encoded)

# Check these
test_df = pd.DataFrame({"orig": y_test.value_counts(), "one_hot": Y_test_onehot.sum(axis=0)})
all(test_df['orig'] == test_df['one_hot'])

model_function=classification_model
model = model_function(
        optimizer=keras_params['optimizer'],
        activation=keras_params['activation'],
        loss=keras_params['loss'],
        sparsity_l1__activity=keras_params['sparsity_l1__activity'],
        sparsity_l2__activity=keras_params['sparsity_l2__activity'],
        sparsity_l1__kernel=keras_params['sparsity_l1__kernel'],
        sparsity_l2__kernel=keras_params['sparsity_l2__kernel'],
        sparsity_l1__bias=keras_params['sparsity_l1__bias'],
        sparsity_l2__bias=keras_params['sparsity_l2__bias']
    )

history = model.fit(
    X_train,
    Y_train_onehot,
    batch_size=batch_size,
    epochs=n_epochs,
    verbose=0,
    # use_multiprocessing=True,
    # validation_split=0.33  # Frac of the training used for validation.
    validation_data=(X_test, Y_test_onehot)
)

# ~~~~~~~~~~~ Check this!!! # ~~~~~~~~~~~ Check this!!! # ~~~~~~~~~~~ Check this!!! 
classes = np.argmax(model.predict(X_test), axis=1)


pred_count = np.unique(classes, return_counts=True)
test_pred = pd.DataFrame({"cluster": pred_count[0], "count": pred_count[1]})

y_test_pred = encoder.inverse_transform(classes) # This converts the standardised classes back into their actual value
y_test_proba = model.predict_proba(X_test)

# underlying function
model_report = pd.DataFrame(classification_report(
        y_test,
        y_test_pred,
        encoder.classes_,
        output_dict=True
    )).transpose()
    
y_pred_proba = y_test_proba
y_true = y_test
y_pred = y_test_pred
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
