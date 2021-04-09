import os
import pickle
import datetime
import pandas as pd
from keras.models import load_model

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


def experiment_plausibility(cfe, covertype, variable_features, metric, continuous_features,
                            categorical_features_lists, bb, X_train, dataset, black_box, filename_results, X_test,
                            nbr_test, continuous_features_all, categorical_features_all, ratio_cont):
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
                        n_attempts=10, n_batch=1000, stopping_eps=0.01, kind='gaussian_matched')

    elif cfe == 'sace-dist-us':
        exp = DistrSACE(variable_features, weights=None, metric=metric,
                        feature_names=None, continuous_features=continuous_features,
                        categorical_features_lists=categorical_features_lists, normalize=False,
                        n_attempts=10, n_batch=1000, stopping_eps=0.01, kind='uniform_sphere')

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
                           n_estimators=n_estimators, max_samples=0.2, max_features='auto',
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
                                        'max_depth': None,
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

    known_train = True
    search_diversity = False
    time_start = datetime.datetime.now()
    exp.fit(bb, X_train)

    y_pred = bb.predict(X_test)

    time_train = (datetime.datetime.now() - time_start).total_seconds()

    index_test_instances = np.random.choice(range(len(X_test)), nbr_test)

    print(datetime.datetime.now(), dataset, black_box, cfe, metric, known_train, search_diversity)

    for test_id, i in enumerate(index_test_instances):

        # if cfe == 'sace-ens-f' and dataset == 'german' and black_box == 'RF' and \
        #         covertype in ['majority',
        #                       'heuristic',
        #                       'naive',
        #                       'naive-sub',
        #                       'knn',
        #                       # 'knn-sub',
        #                       # 'knn-acc',
        #                       # 'knn-acc-sub'
        #                       ]:
        #     continue

        if covertype is None:
            print(datetime.datetime.now(), dataset, black_box, cfe, test_id, len(index_test_instances),
                  '%.2f' % (test_id / len(index_test_instances)))
        else:
            print(datetime.datetime.now(), dataset, black_box, cfe, covertype, test_id, len(index_test_instances),
                  '%.2f' % (test_id / len(index_test_instances)))
        x = X_test[i]
        y_val = bb.predict(x.reshape(1, -1))[0]

        x_eval_list = list()

        for k in [
            1, 2, 3, 4, 5, 8, 10, 12, 14, 16, 18, 20
        ]:
            # print(datetime.datetime.now(), k, '!!!!')
            time_start_i = datetime.datetime.now()

            if '-ens-' not in cfe:
                cf_list = exp.get_counterfactuals(x, k=k, search_diversity=search_diversity)
            else:
                covertype = 'naive' if covertype is None else covertype
                cf_list = exp.get_counterfactuals(x, k=k, search_diversity=search_diversity,
                                                  covertype=covertype,
                                                  lambda_par=1.0, cf_rate=0.5, cf_rate_incr=0.1)

            time_test = (datetime.datetime.now() - time_start_i).total_seconds()

            x_eval = evaluate_only_plasubility(cf_list, x, bb, y_val, k, variable_features, continuous_features_all,
                     categorical_features_all, X_train, X_test, ratio_cont)

            x_eval['dataset'] = dataset
            x_eval['black_box'] = black_box
            x_eval['method'] = cfe
            x_eval['idx'] = test_id
            x_eval['k'] = k
            x_eval['known_train'] = known_train
            x_eval['search_diversity'] = search_diversity
            x_eval['time_train'] = time_train
            x_eval['time_test'] = time_test
            x_eval['runtime'] = time_train + time_test
            x_eval['metric'] = metric if isinstance(metric, str) else '.'.join(metric)
            x_eval['variable_features_flag'] = len(variable_features) > 0
            x_eval['covertype'] = covertype if covertype else 'None'

            x_eval_list.append(x_eval)

        df = pd.DataFrame(data=x_eval_list)
        df = df[['dataset',  'black_box', 'method', 'covertype', 'idx', 'k', 'known_train', 'search_diversity',
                 'metric', 'time_train', 'time_test', 'runtime', 'variable_features_flag', 'nbr_cf', 'nbr_valid_cf',
                 'perc_valid_cf', 'perc_valid_cf_all', 'nbr_actionable_cf', 'perc_actionable_cf',
                 'perc_actionable_cf_all', 'nbr_valid_actionable_cf', 'perc_valid_actionable_cf',
                 'perc_valid_actionable_cf_all', 'plausibility_sum', 'plausibility_max_nbr_cf',
                 'plausibility_nbr_cf', 'plausibility_nbr_valid_cf', 'plausibility_nbr_actionable_cf',
                 'plausibility_nbr_valid_actionable_cf']]

        if not os.path.isfile(filename_results):
            df.to_csv(filename_results, index=False)
        else:
            df.to_csv(filename_results, mode='a', index=False, header=False)


def main():

    dataset = 'compas'
    black_box = 'DNN'
    normalize = 'standard'
    nbr_test = 20

    np.random.seed(random_state)

    if dataset not in dataset_list:
        print('unknown dataset %s' % dataset)
        return -1

    if black_box not in blackbox_list:
        print('unknown black box %s' % black_box)
        return -1

    print(datetime.datetime.now(), dataset, black_box)

    data = get_tabular_dataset(dataset, path_dataset, normalize=normalize, test_size=test_size,
                               random_state=random_state)
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    features_names = data['feature_names']
    variable_features = data['variable_features']
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

    if black_box in ['DT', 'RF', 'SVM', 'NN', 'LGBM']:
        bb = pickle.load(open(path_models + '%s_%s.pickle' % (dataset, black_box), 'rb'))
    elif black_box in ['DNN']:
        bb = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
    else:
        print('unknown black box %s' % black_box)
        raise Exception

    bb = BlackBox(bb)

    metric = ('euclidean', 'jaccard')

    exp = RandomSACE(variable_features, weights=None, metric=metric, feature_names=None,
                     continuous_features=continuous_features, categorical_features_lists=categorical_features_lists,
                     normalize=False, pooler=None, n_attempts=100, n_max_attempts=1000, proba=0.5)
    exp.fit(bb, X_train)

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
            # 'sace-tree-4c',
            # 'sace-tree-4f', 'sace-tree-8c', 'sace-tree-8f', 'sace-tree-16c', 'sace-tree-16f',
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
            # 'sace-ens-r',
        ]:

        if '-ens-' in cfe:
            for covertype in [
                # 'majority',
                #               'heuristic',
                #               'naive',
                #               'naive-sub',
                #               'knn',
                              'knn-sub',
                              # 'knn-acc',
                              # 'knn-acc-sub',
                'knn', 'naive', 'naive-sub', 'majority',
                              ]:
                filename_plusibility = path_results + 'plausibility_%s_%s_%s_%s.csv' % (dataset, black_box, cfe, covertype)
                experiment_plausibility(cfe, covertype, variable_features, metric, continuous_features,
                            categorical_features_lists, bb, X_train, dataset, black_box, filename_plusibility, X_test,
                            nbr_test, continuous_features_all, categorical_features_all, ratio_cont)
        else:
            filename_plusibility = path_results + 'plausibility_%s_%s_%s.csv' % (dataset, black_box, cfe)
            experiment_plausibility(cfe, None, variable_features, metric, continuous_features,
                            categorical_features_lists, bb, X_train, dataset, black_box, filename_plusibility, X_test,
                            nbr_test, continuous_features_all, categorical_features_all, ratio_cont)


if __name__ == "__main__":
    main()


