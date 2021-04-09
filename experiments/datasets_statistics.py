import sys

import os
import pickle
import datetime
import numpy as np
import pandas as pd

from scipy.stats import uniform

import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score

from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

from sace.blackbox import BlackBox

from experiments.config import *
from experiments.util import get_tabular_dataset

def main():
    for dataset in ['adult', 'compas', 'fico', 'german']:

        data = get_tabular_dataset(dataset, path_dataset, normalize='standard', test_size=test_size,
                                   random_state=random_state, encode=None, return_original=False)
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

        class_values = data['class_values']
        if dataset == 'titanic':
            class_values = ['Not Survived', 'Survived']
        features_names = data['feature_names']
        variable_features = data['variable_features']
        variable_features_names = data['variable_features_names']
        continuous_features = data['continuous_features']
        continuous_features_all = data['continuous_features_all']
        categorical_features_lists = data['categorical_features_lists']
        categorical_features_lists_all = data['categorical_features_lists_all']
        categorical_features_all = data['categorical_features_all']
        continuous_features_names = data['continuous_features_names']
        categorical_features_names = data['categorical_features_names']
        scaler = data['scaler']
        nbr_features = data['n_cols']
        ratio_cont = data['n_cont_cols'] / nbr_features
        # dfo = data['dfo']

        data1h = get_tabular_dataset(dataset, path_dataset, normalize='standard', test_size=test_size,
                                   random_state=random_state, encode='onehot', return_original=False)
        one_hot_feat = data1h['feature_names']

        print(dataset, len(X_train) + len(X_test), len(features_names),
              len(continuous_features_all), len(categorical_features_all),
              len(variable_features), len(one_hot_feat), len(class_values))
        for c in sorted(list(set(features_names) - set(variable_features_names))):
            print('\t', c)
        print()
        print()

if __name__ == "__main__":
    main()
