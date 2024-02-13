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


def parse_options():    
    # Inherit options
    parser = argparse.ArgumentParser(
            description="""
                QC of tissues together
                """
        )
    
    parser.add_argument(
        '-t', '--tissue',
        action='store',
        dest='tissue',
        required=True,
        help=''
    )
    
    return parser.parse_args()


def main():
    inherited_options = parse_options()
    tissue = inherited_options.tissue

    print(f"Loaded modules: {tissue}")


# Execute
if __name__ == '__main__':
    main()  
