import sys

import os
import pickle
import datetime
import numpy as np
import pandas as pd

from keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform

from sace.blackbox import BlackBox

from alibi.explainers import CounterFactual

from cf_eval.metrics import *

from experiments.config import *
from experiments.util import get_tabular_dataset

from sace.random_sace import RandomSACE

# Watcher method


def experiment(cfe, bb, X_train, variable_features, metric, continuous_features, categorical_features_lists,
               X_test, nbr_test, search_diversity, dataset, black_box, known_train, continuous_features_all,
               categorical_features_all, ratio_cont, nbr_features, filename_results, variable_features_flag,
               features_names, couples_to_test,
               exp_calc_dist):

    time_start = datetime.datetime.now()
    shape = (1,) + X_train.shape[1:]
    target_proba = 0.5
    tol = 0.01  # want counterfactuals with p(class)>0.99
    target_class = 'other'
    max_iter = 100
    lam_init = 0.1
    max_lam_steps = 10
    learning_rate_init = 0.1

    predict_fn = lambda x: bb.predict_proba(x)

    if not variable_features_flag:
        feature_range = (X_train.min(axis=0).reshape(shape),  # feature range for the perturbed instance
                         X_train.max(axis=0).reshape(shape))  # can be either a float or array of shape (1xfeatures)

        # initialize explainer
        exp = CounterFactual(predict_fn, shape=shape, target_proba=target_proba, tol=tol,
                            target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                            max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                            feature_range=feature_range)

    time_train = (datetime.datetime.now() - time_start).total_seconds()

    index_test_instances = np.random.choice(range(len(X_test)), nbr_test)

    print(datetime.datetime.now(), dataset, black_box, cfe, 'instability')

    for test_id, couple in enumerate(couples_to_test):
        print(datetime.datetime.now(), dataset, black_box, cfe, test_id, len(couples_to_test),
              '%.2f' % (test_id / len(couples_to_test)), 'instability')
        x1 = couple[0]
        x2 = couple[1]
        x_eval_list = list()

        time_start_i = datetime.datetime.now()
        if variable_features_flag:
            feature_range = (X_train.min(axis=0).reshape(shape),  # feature range for the perturbed instance
                             X_train.max(axis=0).reshape(shape))  # can be either a float or array of shape (1xfeatures)

            feature_range[0][:, variable_features] = x1[variable_features]
            feature_range[1][:, variable_features] = x1[variable_features]

            # initialize explainer
            exp1 = CounterFactual(predict_fn, shape=shape, target_proba=target_proba, tol=tol,
                                target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                                max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                                feature_range=feature_range)

            feature_range[0][:, variable_features] = x2[variable_features]
            feature_range[1][:, variable_features] = x2[variable_features]

            # initialize explainer
            exp2 = CounterFactual(predict_fn, shape=shape, target_proba=target_proba, tol=tol,
                                  target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                                  max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                                  feature_range=feature_range)

        explanation1 = exp1.explain(x1.reshape(1, -1))
        if explanation1.cf is not None:
            cf_list1 = explanation1.cf['X']
        else:
            cf_list1 = np.array([])

        explanation2 = exp2.explain(x1.reshape(1, -1))
        if explanation2.cf is not None:
            cf_list2 = explanation2.cf['X']
        else:
            cf_list2 = np.array([])

        time_test = (datetime.datetime.now() - time_start_i).total_seconds()

        for k in [1, 2, 3, 4, 5, 8, 10, 12, 14, 16, 18, 20]:
            d_x1x2 = exp_calc_dist.cdist(x1.reshape(1, -1), x2.reshape(1, -1))[0][0]

            sum_c1c2 = 0.0
            for c1 in cf_list1:
                for c2 in cf_list2:
                    d_c1c2 = exp_calc_dist.cdist(c1.reshape(1, -1), c2.reshape(1, -1))[0][0]
                    sum_c1c2 += d_c1c2

            if len(cf_list1) > 0 and len(cf_list2) > 0:
                inst_x1x2 = 1.0 / (1.0 + d_x1x2) * 1.0 / (len(cf_list1) * len(cf_list2)) * sum_c1c2
            else:
                inst_x1x2 = np.nan

            x_eval = dict()
            x_eval['inst_x1x2'] = inst_x1x2

            x_eval['dataset'] = dataset
            x_eval['black_box'] = black_box
            x_eval['method'] = cfe
            x_eval['couple_idx'] = test_id
            x_eval['k'] = k
            x_eval['known_train'] = known_train
            x_eval['search_diversity'] = search_diversity
            x_eval['time_train'] = time_train
            x_eval['time_test'] = time_test
            x_eval['runtime'] = time_train + time_test
            x_eval['metric'] = metric if isinstance(metric, str) else '.'.join(metric)
            x_eval['variable_features_flag'] = len(variable_features) > 0

            x_eval_list.append(x_eval)

        df = pd.DataFrame(data=x_eval_list)
        df = df[['dataset', 'black_box', 'method', 'couple_idx', 'k', 'known_train', 'search_diversity',
                 'metric', 'time_train', 'time_test', 'runtime', 'variable_features_flag', 'inst_x1x2']]

        if not os.path.isfile(filename_results):
            df.to_csv(filename_results, index=False)
        else:
            df.to_csv(filename_results, mode='a', index=False, header=False)


def main():

    nbr_test = 20
    dataset = 'compas'
    black_box = 'RF'
    normalize = 'standard'

    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()

    # dataset = sys.argv[1]
    # black_box = sys.argv[2]
    # cfe = sys.argv[3]
    # nbr_test = 100 if len(sys.argv) < 5 else int(sys.argv[4])
    known_train = True
    search_diversity = False
    metric = 'none'
    variable_features_flag = True
    nbr_exp = 20

    np.random.seed(random_state)

    if dataset not in dataset_list:
        print('unknown dataset %s' % dataset)
        return -1

    if black_box not in blackbox_list:
        print('unknown black box %s' % black_box)
        return -1

    # if cfe not in cfe_list:
    #     print('unknown counterfactual explainer %s' % cfe)
    #     return -1

    print(datetime.datetime.now(), dataset, black_box)

    data = get_tabular_dataset(dataset, path_dataset, normalize=normalize, test_size=test_size,
                               random_state=random_state, encode=None)
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

    variable_cont_features_names = [c for c in variable_features_names if c in continuous_features_names]
    variable_cate_features_names = list(
        set([c.split('=')[0] for c in variable_features_names if c.split('=')[0] in categorical_features_names]))

    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    if black_box in ['DT', 'RF', 'SVM', 'NN', 'LGBM']:
        bb = pickle.load(open(path_models + '%s_%s.pickle' % (dataset, black_box), 'rb'))
        # if black_box == 'RF':
        #     bb.n_jobs = 5
    elif black_box in ['DNN']:

        bb = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
    else:
        print('unknown black box %s' % black_box)
        raise Exception

    bb = BlackBox(bb)

    metric_calc_dist = ('euclidean', 'jaccard')

    exp_calc_dist = RandomSACE(variable_features, weights=None, metric=metric_calc_dist, feature_names=None,
                               continuous_features=continuous_features,
                               categorical_features_lists=categorical_features_lists,
                               normalize=False, pooler=None, n_attempts=100, n_max_attempts=1000, proba=0.5)
    exp_calc_dist.fit(bb, X_train)

    y_pred = bb.predict(X_test)
    class_values = sorted(np.unique(y_pred))
    nbr_exp_per_class = nbr_exp // len(class_values)

    couples_to_test = list()
    for class_val in class_values:
        X_test_y = X_test[y_pred == class_val]
        for i, x in enumerate(X_test_y):
            neigh_dist = exp_calc_dist.cdist(x.reshape(1, -1), X_test_y)
            idx_neigh = np.argsort(neigh_dist, kind='stable')[0]
            closest_idx = idx_neigh[0] if i != idx_neigh[0] else idx_neigh[1]
            couples_to_test.append((x, X_test_y[closest_idx]))
            if i >= nbr_exp_per_class:
                break

    filename_stability = path_results + 'instability_%s_%s_cfw.csv' % (dataset, black_box)

    experiment('cfw', bb, X_train, variable_features, metric,
               continuous_features, categorical_features_lists,
               X_test, nbr_test, search_diversity, dataset, black_box, known_train,
               continuous_features_all, categorical_features_all, ratio_cont, nbr_features,
               filename_stability, variable_features_flag, features_names, couples_to_test, exp_calc_dist)


if __name__ == "__main__":
    main()


