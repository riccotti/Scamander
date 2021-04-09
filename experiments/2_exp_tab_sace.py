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
from sace.random_sace import RandomSACE
from sace.feature_sace import FeatureSACE
from sace.neighbor_sace import NeighborSACE
from sace.cluster_sace import KMeansSACE
from sace.tree_sace import TreeSACE
from sace.ensemble_sace import EnsembleSACE
from sace.distr_sace import DistrSACE
from sace.casebased_sace import CaseBasedSACE

from cf_eval.metrics import *

from experiments.config import *
from experiments.util import get_tabular_dataset


def experiment(cfe, bb, X_train, variable_features, metric, continuous_features, categorical_features_lists,
               X_test, nbr_test, search_diversity, dataset, black_box, known_train, continuous_features_all,
               categorical_features_all, ratio_cont, nbr_features, filename_results, filename_cf, features_names,
               covertype=None):

    time_start = datetime.datetime.now()
    n_estimators = 10
    if cfe == 'sace-rand-1000':
        exp = RandomSACE(variable_features, weights=None, metric=metric,
                         feature_names=None, continuous_features=continuous_features,
                         categorical_features_lists=categorical_features_lists, normalize=False,
                         n_attempts=100, n_max_attempts=1000, proba=0.5)
    elif cfe == 'sace-rand-100000':
        exp = RandomSACE(variable_features, weights=None, metric=metric,
                         feature_names=None, continuous_features=continuous_features,
                         categorical_features_lists=categorical_features_lists, normalize=False,
                         n_attempts=1000, n_max_attempts=100000, proba=0.5)
    elif cfe == 'sace-feat-1-5':
        exp = FeatureSACE(variable_features, weights=None, metric=metric,
                          feature_names=None, continuous_features=continuous_features,
                          categorical_features_lists=categorical_features_lists, normalize=False,
                          nbr_intervals=5, nbr_features_to_test=1)
    elif cfe == 'sace-feat-1-10':
        exp = FeatureSACE(variable_features, weights=None, metric=metric,
                          feature_names=None, continuous_features=continuous_features,
                          categorical_features_lists=categorical_features_lists, normalize=False,
                          nbr_intervals=10, nbr_features_to_test=1)
    elif cfe == 'sace-feat-1-20':
        exp = FeatureSACE(variable_features, weights=None, metric=metric,
                          feature_names=None, continuous_features=continuous_features,
                          categorical_features_lists=categorical_features_lists, normalize=False,
                          nbr_intervals=20, nbr_features_to_test=1)
    elif cfe == 'sace-feat-2-5':
        exp = FeatureSACE(variable_features, weights=None, metric=metric,
                          feature_names=None, continuous_features=continuous_features,
                          categorical_features_lists=categorical_features_lists, normalize=False,
                          nbr_intervals=5, nbr_features_to_test=2)
    elif cfe == 'sace-feat-2-10':
        exp = FeatureSACE(variable_features, weights=None, metric=metric,
                          feature_names=None, continuous_features=continuous_features,
                          categorical_features_lists=categorical_features_lists, normalize=False,
                          nbr_intervals=10, nbr_features_to_test=2)
    elif cfe == 'sace-feat-2-20':
        exp = FeatureSACE(variable_features, weights=None, metric=metric,
                          feature_names=None, continuous_features=continuous_features,
                          categorical_features_lists=categorical_features_lists, normalize=False,
                          nbr_intervals=20, nbr_features_to_test=2)
    elif cfe == 'sace-neig-all':
        exp = NeighborSACE(variable_features, weights=None, metric=metric,
                           feature_names=None, continuous_features=continuous_features,
                           categorical_features_lists=categorical_features_lists, normalize=False, random_samples=None)
    elif cfe == 'sace-neig-100':
        exp = NeighborSACE(variable_features, weights=None, metric=metric,
                           feature_names=None, continuous_features=continuous_features,
                           categorical_features_lists=categorical_features_lists, normalize=False, random_samples=100)
    elif cfe == 'sace-clus-10':
        exp = KMeansSACE(variable_features, weights=None, metric=metric,
                         feature_names=None, continuous_features=continuous_features,
                         categorical_features_lists=categorical_features_lists, normalize=False,
                         n_clusters=10, init='k-means++', n_init=10, max_iter=300, tol=0.0001)
    elif cfe == 'sace-clus-100':
        exp = KMeansSACE(variable_features, weights=None, metric=metric,
                         feature_names=None, continuous_features=continuous_features,
                         categorical_features_lists=categorical_features_lists, normalize=False,
                         n_clusters=100, init='k-means++', n_init=10, max_iter=300, tol=0.0001)
    elif cfe == 'sace-tree-infc':
        exp = TreeSACE(variable_features, weights=None, metric=metric,
                       feature_names=None, continuous_features=continuous_features,
                       categorical_features_lists=categorical_features_lists, normalize=False,
                       use_instance_weights=True, kernel_width=None,
                       min_samples_leaf=1, max_depth=None, closest_in_leaf=True)
    elif cfe == 'sace-tree-16c':
        exp = TreeSACE(variable_features, weights=None, metric=metric,
                       feature_names=None, continuous_features=continuous_features,
                       categorical_features_lists=categorical_features_lists, normalize=False,
                       use_instance_weights=True, kernel_width=None,
                       min_samples_leaf=0.01, max_depth=16, closest_in_leaf=True)
    elif cfe == 'sace-tree-16f':
        exp = TreeSACE(variable_features, weights=None, metric=metric,
                       feature_names=None, continuous_features=continuous_features,
                       categorical_features_lists=categorical_features_lists, normalize=False,
                       use_instance_weights=True, kernel_width=None,
                       min_samples_leaf=0.01, max_depth=16, closest_in_leaf=False)
    elif cfe == 'sace-tree-8c':
        exp = TreeSACE(variable_features, weights=None, metric=metric,
                       feature_names=None, continuous_features=continuous_features,
                       categorical_features_lists=categorical_features_lists, normalize=False,
                       use_instance_weights=True, kernel_width=None,
                       min_samples_leaf=0.01, max_depth=8, closest_in_leaf=True)
    elif cfe == 'sace-tree-8f':
        exp = TreeSACE(variable_features, weights=None, metric=metric,
                       feature_names=None, continuous_features=continuous_features,
                       categorical_features_lists=categorical_features_lists, normalize=False,
                       use_instance_weights=True, kernel_width=None,
                       min_samples_leaf=0.01, max_depth=8, closest_in_leaf=False)
    elif cfe == 'sace-tree-4c':
        exp = TreeSACE(variable_features, weights=None, metric=metric,
                       feature_names=None, continuous_features=continuous_features,
                       categorical_features_lists=categorical_features_lists, normalize=False,
                       use_instance_weights=True, kernel_width=None,
                       min_samples_leaf=0.01, max_depth=4, closest_in_leaf=True)
    elif cfe == 'sace-tree-4f':
        exp = TreeSACE(variable_features, weights=None, metric=metric,
                       feature_names=None, continuous_features=continuous_features,
                       categorical_features_lists=categorical_features_lists, normalize=False,
                       use_instance_weights=True, kernel_width=None,
                       min_samples_leaf=0.01, max_depth=4, closest_in_leaf=False)

    elif cfe == 'sace-cb':
        exp = CaseBasedSACE(variable_features, weights=None, metric=metric,
                            feature_names=None, continuous_features=continuous_features,
                            categorical_features_lists=categorical_features_lists, normalize=False,
                            random_samples=None, diff_features=2, tolerance=0.001)

    elif cfe == 'sace-dist-gm':
        exp = DistrSACE(variable_features, weights=None, metric=metric,
                        feature_names=None, continuous_features=continuous_features,
                        categorical_features_lists=categorical_features_lists, normalize=False,
                        n_attempts=10000, n_batch=1000, stopping_eps=0.01, kind='gaussian_matched')

    elif cfe == 'sace-dist-us':
        exp = DistrSACE(variable_features, weights=None, metric=metric,
                        feature_names=None, continuous_features=continuous_features,
                        categorical_features_lists=categorical_features_lists, normalize=False,
                        n_attempts=10000, n_batch=1000, stopping_eps=0.01, kind='uniform_sphere')

    elif cfe == 'sace-ens-d':
        exp = EnsembleSACE(variable_features, weights=None, metric=metric,
                           feature_names=None, continuous_features=continuous_features,
                           categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                           base_estimator='dist', n_estimators=n_estimators, max_samples=0.2, max_features='auto',
                           n_jobs=-1, verbose=0)
    elif cfe == 'sace-ens-t':
        exp = EnsembleSACE(variable_features, weights=None, metric=metric,
                           feature_names=None, continuous_features=continuous_features,
                           categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                           base_estimator='tree', n_estimators=n_estimators, max_samples=0.2, max_features='auto',
                           n_jobs=-1, verbose=0)
    elif cfe == 'sace-ens-f':
        exp = EnsembleSACE(variable_features, weights=None, metric=metric,
                           feature_names=None, continuous_features=continuous_features,
                           categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                           base_estimator='feat', n_estimators=n_estimators, max_samples=0.2, max_features='auto',
                           n_jobs=-1, verbose=0)
    elif cfe == 'sace-ens-n':
        exp = EnsembleSACE(variable_features, weights=None, metric=metric,
                           feature_names=None, continuous_features=continuous_features,
                           categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                           base_estimator='neigh', n_estimators=n_estimators, max_samples=0.2, max_features='auto',
                           n_jobs=-1, verbose=0)
    elif cfe == 'sace-ens-c':
        exp = EnsembleSACE(variable_features, weights=None, metric=metric,
                           feature_names=None, continuous_features=continuous_features,
                           categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                           base_estimator='cb', n_estimators=n_estimators, max_samples=0.2, max_features='auto',
                           n_jobs=-1, verbose=0)
    elif cfe == 'sace-ens-l':
        exp = EnsembleSACE(variable_features, weights=None, metric=metric,
                           feature_names=None, continuous_features=continuous_features,
                           categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                           base_estimator='clus', n_estimators=n_estimators, max_samples=0.2, max_features='auto',
                           n_jobs=-1, verbose=0)
    elif cfe == 'sace-ens-r':
        exp = EnsembleSACE(variable_features, weights=None, metric=metric,
                           feature_names=None, continuous_features=continuous_features,
                           categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                           base_estimator='rand', n_estimators=n_estimators, max_samples=0.2, max_features='auto',
                           n_jobs=-1, verbose=0)
    elif cfe == 'sace-ens-h':
        exp = EnsembleSACE(variable_features, weights=None, metric=metric,
                           feature_names=None, continuous_features=continuous_features,
                           categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                           n_estimators=n_estimators, max_samples=0.01, max_features='auto',
                           base_estimator='pippo',
                           estimators_params={
                               'dist': {'n_attempts': 10,
                                        'n_batch': 1000,
                                        'stopping_eps': 0.01,
                                        'kind': 'gaussian_matched',
                                        'tol': 0.01},
                               'tree': {'use_instance_weights': False,
                                        'kernel_width': None,
                                        'min_samples_leaf': 0.01,
                                        'max_depth': None,  #16
                                        'closest_in_leaf': True},
                               'feat': {'nbr_intervals': 10,
                                        'nbr_features_to_test': 1,
                                        'tol': 0.01},
                               # 'neig': {'random_samples': 100},
                               # 'rand': {}
                               # 'cb': {},
                               # 'clus': {},
                           },
                           n_jobs=-1, verbose=0)
    else:
        print('unknown counterfactual explainer %s' % cfe)
        raise Exception

    exp.fit(bb, X_train)

    # for i, e in enumerate(exp.estimators_):
    #     print(i, e.variable_features)

    time_train = (datetime.datetime.now() - time_start).total_seconds()

    index_test_instances = np.random.choice(range(len(X_test)), nbr_test)

    print(datetime.datetime.now(), dataset, black_box, cfe, metric, known_train, search_diversity)

    for test_id, i in enumerate(index_test_instances):

        if cfe == 'sace-ens-f' and dataset == 'german' and black_box == 'RF' and \
            covertype in ['majority',
                          'heuristic',
                          'naive',
                          'naive-sub',
                          'knn',
                          # 'knn-sub',
                          # 'knn-acc',
                          # 'knn-acc-sub'
        ]:
            continue

        # if cfe == 'sace-ens-h' and dataset == 'compas' and black_box == 'RF' and test_id < 16\
        #         and covertype == 'knn-sub':
        #     continue

        # if cfe == 'sace-ens-h' and dataset == 'compas' and black_box == 'RF' and covertype == 'majority':
        #     continue
        #
        # if cfe == 'sace-dist-gm' and dataset == 'german' and black_box == 'DNN' and test_id <= 62:
        #     continue
        #
        # if cfe == 'sace-rand-1000' and dataset == 'german' and black_box == 'DNN' and test_id < 12:
        #     continue

        # if cfe == 'sace-cb' and dataset == 'german' and black_box == 'DNN' and test_id < 81:
        # if cfe == 'sace-feat-2-10' and test_id < 39:
        #     if metric in [
        #         ('euclidean', 'jaccard'),
        #         ('cosine', 'jaccard'),
        #         # ('euclidean', 'hamming')
        #     ]:
        #         return
        #     continue

        if covertype is None:
            print(datetime.datetime.now(), dataset, black_box, cfe, test_id, len(index_test_instances),
                  '%.2f' % (test_id / len(index_test_instances)))
        else:
            print(datetime.datetime.now(), dataset, black_box, cfe, covertype, test_id, len(index_test_instances),
                  '%.2f' % (test_id / len(index_test_instances)))
        x = X_test[i]
        y_val = bb.predict(x.reshape(1, -1))[0]

        cf_list_all = list()

        x_eval_list = list()

        for k in [
            1, 2, 3, 4, 5, 8, 10, 12, 14, 16, 18, 20
            ]:
            # print(datetime.datetime.now(), k, '!!!!')
            time_start_i = datetime.datetime.now()

            if '-ens-' not in cfe:
                cf_list = exp.get_counterfactuals(x, k=k)  #, search_diversity=search_diversity)
            else:
                covertype = 'naive' if covertype is None else covertype
                cf_list = exp.get_counterfactuals(x, k=k, search_diversity=search_diversity,
                                                  covertype=covertype,
                                                  lambda_par=1.0, cf_rate=0.5, cf_rate_incr=0.1)

            # print(cf_list)
            # return -1

            time_test = (datetime.datetime.now() - time_start_i).total_seconds()

            x_eval = evaluate_cf_list(cf_list, x, bb, y_val, k, variable_features,
                                      continuous_features_all, categorical_features_all, X_train, X_test,
                                      ratio_cont, nbr_features)

            x_eval['dataset'] = dataset
            x_eval['black_box'] = black_box
            x_eval['method'] = cfe
            x_eval['idx'] = i
            x_eval['k'] = k
            x_eval['known_train'] = known_train
            x_eval['search_diversity'] = search_diversity
            x_eval['time_train'] = time_train
            x_eval['time_test'] = time_test
            x_eval['runtime'] = time_train + time_test
            x_eval['metric'] = metric if isinstance(metric, str) else '.'.join(metric)
            x_eval['variable_features_flag'] = len(variable_features) > 0

            x_eval_list.append(x_eval)
            if len(cf_list):
                cf_list_all.append(cf_list[0])

        if len(cf_list_all) > 1:
            instability_si = np.mean(squareform(pdist(np.array(cf_list_all), metric='euclidean')))
        else:
            instability_si = 0.0

        for x_eval in x_eval_list:
            x_eval['instability_si'] = instability_si

        df = pd.DataFrame(data=x_eval_list)
        df = df[columns]

        if not os.path.isfile(filename_results):
            df.to_csv(filename_results, index=False)
        else:
            df.to_csv(filename_results, mode='a', index=False, header=False)

        df_cf = pd.DataFrame(data=cf_list_all, columns=features_names)
        df_cf['idx'] = [i] * len(cf_list_all)
        df_cf['test_id'] = np.arange(0, len(cf_list_all))
        df_cf['dataset'] = [dataset] * len(cf_list_all)
        df_cf['black_box'] = [black_box] * len(cf_list_all)
        df_cf['method'] = [cfe] * len(cf_list_all)
        df_cf['known_train'] = [known_train] * len(cf_list_all)
        df_cf['search_diversity'] = [search_diversity] * len(cf_list_all)

        if not os.path.isfile(filename_cf):
            df_cf.to_csv(filename_cf, index=False)
        else:
            df_cf.to_csv(filename_cf, mode='a', index=False, header=False)


def main():

    nbr_test = 100
    dataset = 'german'
    black_box = 'RF'
    normalize = 'standard'
    # normalize_str = '' if normalize is None else '_%s' % normalize
    # dataset = sys.argv[1]
    # black_box = sys.argv[2]
    # cfe = sys.argv[3]
    # nbr_test = 100 if len(sys.argv) < 5 else int(sys.argv[4])
    # known_train = True if len(sys.argv) < 6 else sys.argv[5]
    # search_diversity = True

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
                               random_state=random_state, encode=None if black_box == 'LGBM' else 'onehot')
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

    if black_box in ['DT', 'RF', 'SVM', 'NN', 'LGBM']:
        bb = pickle.load(open(path_models + '%s_%s.pickle' % (dataset, black_box), 'rb'))
    elif black_box in ['DNN']:
        bb = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
    else:
        print('unknown black box %s' % black_box)
        raise Exception

    bb = BlackBox(bb)

    for known_train in [True]:  # [True, False]:

        # if known_train:
        #     X_train = X_train
        #     X_test = X_test
        # else:
        #     X_train, X_test, _, _ = train_test_split(X_test, y_test, test_size=test_size,
        #                                              random_state=random_state, stratify=y_test)

        for cfe in [
            # 'sace-dist-gm',
            # 'sace-dist-us',
            # 'sace-cb',
            # 'sace-feat-1-5',
            # 'sace-feat-1-10',
            # 'sace-feat-1-20',
            # 'sace-feat-2-5',
            # 'sace-feat-2-10',
            # 'sace-feat-2-20',
            # 'sace-neig-100', 'sace-neig-all',
            # 'sace-clus-10', 'sace-clus-100',
            # 'sace-tree-4c', 'sace-tree-4f', 'sace-tree-8c', 'sace-tree-8f', 'sace-tree-16c', 'sace-tree-16f',
            # 'sace-tree-infc',
            # 'sace-rand-1000',
            # 'sace-rand-100000',
            # 'sace-ens',
            'sace-ens-h',
            'sace-ens-d',
            'sace-ens-t',
            'sace-ens-f',
            'sace-ens-n',
            # 'sace-ens-c',
            # 'sace-ens-l',
            'sace-ens-r',
        ]:

            filename_results = path_results + 'cfeval_%s_%s_%s.csv' % (dataset, black_box, cfe)
            filename_cf = path_cf + 'cf_%s_%s_%s.csv' % (dataset, black_box, cfe)

            for search_diversity in [False]:  # [True, False]:

                for metric in [
                    ('euclidean', 'jaccard'),
                    #('cosine', 'jaccard'), ('euclidean', 'hamming'), 'euclidean'
                ]:

                    if 'tree' in cfe and 'cosine' in metric:
                        continue

                    if '-ens-' not in cfe:
                        experiment(cfe, bb, X_train, variable_features, metric,
                                   continuous_features, categorical_features_lists,
                                   X_test, nbr_test, search_diversity, dataset, black_box, known_train,
                                   continuous_features_all, categorical_features_all, ratio_cont, nbr_features,
                                   filename_results, filename_cf, features_names)
                    else:
                        for covertype in ['majority',
                                          'heuristic',
                                          'naive',
                                          'naive-sub',
                                          'knn',
                                          'knn-sub',
                                          'knn-acc',
                                          'knn-acc-sub'
                                          ]:

                            cfe_str = cfe + '_' + covertype
                            filename_results = path_results + 'cfeval_%s_%s_%s.csv' % (dataset, black_box, cfe_str)
                            filename_cf = path_cf + 'cf_%s_%s_%s.csv' % (dataset, black_box, cfe_str)

                            experiment(cfe, bb, X_train, variable_features, metric,
                                       continuous_features, categorical_features_lists,
                                       X_test, nbr_test, search_diversity, dataset, black_box, known_train,
                                       continuous_features_all, categorical_features_all, ratio_cont, nbr_features,
                                       filename_results, filename_cf, features_names, covertype)


if __name__ == "__main__":
    main()


