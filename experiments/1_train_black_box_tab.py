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


params = {
    'RF': {
        'n_estimators': [8, 16, 32, 64, 128, 256, 512, 1024],
        'min_samples_split': [2, 0.002, 0.01, 0.05, 0.1, 0.2],
        'min_samples_leaf': [1, 0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [None, 2, 4, 6, 8, 10, 12, 16],
        'class_weight': [None, 'balanced'],
        'random_state': [0],
    },
    'NN': {
        'hidden_layer_sizes': [(4,), (8,), (16,), (32,), (64,), (64, 16,), (128, 64, 8,)],
        'activation': ['logistic', 'tanh', 'relu'],
        'alpha': uniform(0.001, 0.1),
        'learning_rate': ['constant'],
        'learning_rate_init': uniform(0.001, 0.1),
        'max_iter': [10000],
        'random_state': [0],
    },
    'SVM': {
        'C': uniform(0.01, 1.0),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': uniform(0.01, 0.1),
        'coef0': uniform(0.01, 0.1),
        'class_weight': [None, 'balanced'],
        'max_iter': [1000],
        'random_state': [0],
    },
    'DNN': {
        'activation_0': ['sigmoid', 'tanh', 'relu'],
        'activation_1': ['sigmoid', 'tanh', 'relu'],
        'activation_2': ['sigmoid', 'tanh', 'relu'],
        'dim_1': [1024, 512, 256, 128, 64, 32, 16, 8, 4],
        'dim_2': [1024, 512, 256, 128, 64, 32, 16, 8, 4],
        'dropout_0': [None, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01],
        'dropout_1': [None, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01],
        'dropout_2': [None, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01],
        'optimizer': ['adam', 'rmsprop', 'sgd'],
    },
    'LGBM': {
        'boosting_type': ['gbdt'],
        'num_leaves': [4, 8, 16, 32, 64, 128],
        'max_depth': [-1, 2, 4, 6, 8, 10, 12, 16],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'n_estimators': [8, 16, 32, 64, 128, 256, 512, 1024],
        'random_state': [0],
    }
}


logboard = TensorBoard(log_dir='.logs', histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch')


def build_dnn(dim_0, dim_1, dim_2, activation_0, activation_1, activation_2, dropout_0, dropout_1, dropout_2,
              optimizer, loss, dim_out):
    model = Sequential()

    model.add(Dense(dim_0, activation=activation_0, kernel_initializer='uniform'))
    if dropout_0 is not None:
        model.add(Dropout(dropout_0))

    model.add(Dense(dim_1, activation=activation_1, kernel_initializer='uniform'))
    if dropout_1 is not None:
        model.add(Dropout(dropout_1))

    model.add(Dense(dim_2, activation=activation_2, kernel_initializer='uniform'))  # uniform, random_normal
    if dropout_2 is not None:
        model.add(Dropout(dropout_2))

    model.add(Dense(dim_out, activation='sigmoid'))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model


def build_dnn_laura(dim_input, dim_1, dim_2, dim_3, dropout_0, dropout_1, dropout_2, dropout_3):
    model = Sequential()

    model.add(Dense(dim_input, activation='relu', kernel_initializer='uniform'))
    if dropout_0 is not None:
        model.add(Dropout(dropout_0))

    model.add(Dense(dim_1, activation='relu', kernel_initializer='uniform'))
    if dropout_1 is not None:
        model.add(Dropout(dropout_1))

    model.add(Dense(dim_2, activation='relu', kernel_initializer='uniform'))  # uniform, random_normal
    if dropout_2 is not None:
        model.add(Dropout(dropout_2))

    if dim_3 is not None:
        model.add(Dense(dim_3, activation='relu', kernel_initializer='uniform'))  # uniform, random_normal
        if dropout_3 is not None:
            model.add(Dropout(dropout_3))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# def build_dnn(dim_in, activation_in, dim_h, activation_h, optimizer, loss, dim_out):
#     model = Sequential()
#
#     model.add(Dense(dim_in, activation=activation_in))
#     model.add(Dense(dim_h, activation=activation_h))
#     model.add(Dense(dim_out, activation='sigmoid'))
#     model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
#
#     return model


def main():

    # dataset = sys.argv[1]
    # black_box = sys.argv[2]
    # dataset = 'bank'
    # black_box = 'SVM'

    black_box = 'DNN'
    for dataset in [
        'compas'
        # 'adult',
        #             #'bank', 'churn',
        # 'compas', #'diabetes',
        # 'fico',
        # 'german',
        # 'home', 'titanic'
    ]:

        normalize = 'standard'
        # normalize_str = '' if normalize is None else '_%s' % normalize
        n_iter = 100

        if dataset not in dataset_list:
            print('unknown dataset %s' % dataset)
            return -1

        if black_box not in blackbox_list:
            print('unknown black box %s' % black_box)
            return -1

        print(datetime.datetime.now(), dataset, black_box)

        data = get_tabular_dataset(dataset, path_dataset, normalize=normalize, test_size=test_size,
                                   random_state=random_state, encode=None if black_box == 'LGBM' else 'onehot')
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

        if black_box in ['RF', 'SVM', 'NN', 'LGBM']:
            n_jobs = 5
            if black_box == 'RF':
                bb = RandomForestClassifier()
            elif black_box == 'SVM':
                bb = SVC(probability=True)
            elif black_box == 'NN':
                bb = MLPClassifier()
            elif black_box == 'LGBM':
                bb = lgb.LGBMClassifier()
            else:
                print('unknown black box %s' % black_box)
                raise Exception

            rs = RandomizedSearchCV(bb, param_distributions=params[black_box], n_iter=n_iter, cv=5, scoring='f1_macro',
                                    iid=False, n_jobs=n_jobs, verbose=1)
            rs.fit(X_train, y_train)
            bb = rs.best_estimator_

        elif black_box == 'DNN':
            dim_0 = X_train.shape[1]
            if len(np.unique(y_train)) == 2:
                loss = 'binary_crossentropy'
                dim_out = 1
            else:
                loss = 'categorical_crossentropy'
                dim_out = len(np.unique(y_train))

            if len(np.unique(y_train)) > 2:
                encoder = OneHotEncoder()
                y_train1h = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
            else:
                y_train1h = y_train

            bb = build_dnn(dim_0=dim_0, dim_1=128, dim_2=64,
                           activation_0='relu', activation_1='relu', activation_2='relu',
                           dropout_0=0.3, dropout_1=0.1, dropout_2=0.01, optimizer='adam', loss=loss, dim_out=dim_out)

            X_train_dnn, X_val_dnn, y_train_dnn, y_val_dnn = train_test_split(X_train, y_train1h, test_size=test_size,
                                                                              random_state=random_state, stratify=y_train1h)

            bb.fit(X_train_dnn, y_train_dnn, validation_data=(X_val_dnn, y_val_dnn), epochs=100, batch_size=128)
        else:
            print('unknown black box %s' % black_box)
            raise Exception

        if black_box == 'DNN':
            bb.save(path_models + '%s_%s.h5' % (dataset, black_box))
        else:
            pickle_file = open(path_models + '%s_%s.pickle' % (dataset, black_box), 'wb')
            pickle.dump(bb, pickle_file)
            pickle_file.close()

        bb = BlackBox(bb)

        y_pred_train = bb.predict(X_train)
        y_pred_test = bb.predict(X_test)

        res = {
            'dataset': dataset,
            'black_box': black_box,
            'accuracy_train': accuracy_score(y_train, y_pred_train),
            'accuracy_test': accuracy_score(y_test, y_pred_test),
            'f1_macro_train': f1_score(y_train, y_pred_train, average='macro'),
            'f1_macro_test': f1_score(y_test, y_pred_test, average='macro'),
            'f1_micro_train': f1_score(y_train, y_pred_train, average='micro'),
            'f1_micro_test': f1_score(y_test, y_pred_test, average='micro'),
        }

        print(dataset, black_box)
        print('accuracy_train', res['accuracy_train'])
        print('accuracy_test', res['accuracy_test'])
        print(np.unique(bb.predict(X_test), return_counts=True))

        df = pd.DataFrame(data=[res])
        columns = ['dataset', 'black_box', 'accuracy_train', 'accuracy_test', 'f1_macro_train', 'f1_macro_test',
                   'f1_micro_train', 'f1_micro_test']
        df = df[columns]

        filename_results = path_results + 'classifiers_performance.csv'
        if not os.path.isfile(filename_results):
            df.to_csv(filename_results, index=False)
        else:
            df.to_csv(filename_results, mode='a', index=False, header=False)


if __name__ == "__main__":
    main()

