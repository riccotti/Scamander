import sys

import os
import pickle
import datetime
import numpy as np
import pandas as pd

from keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform

import dice_ml

from cf_eval.metrics import *
from sace.blackbox import BlackBox

from experiments.config import *
from experiments.util import get_tabular_dataset

from sace.random_sace import RandomSACE


def experiment(cfe, bb, bb_dice, d, X_train, variable_features, features_names, variable_features_names,
               X_test, nbr_test, dataset, black_box, continuous_features_all,
               categorical_features_all, ratio_cont, nbr_features, filename_results,
               variable_features_flag, known_train, search_diversity, metric, backend, couples_to_test, exp_calc_dist):

    time_start = datetime.datetime.now()
    m = dice_ml.Model(model=bb_dice, backend=backend)
    try:
        exp = dice_ml.Dice(d, m)
    except:
        pass

    bb_dice = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
    bb = BlackBox(bb_dice)
    m = dice_ml.Model(model=bb_dice, backend=backend)
    exp = dice_ml.Dice(d, m)

    time_train = (datetime.datetime.now() - time_start).total_seconds()

    if variable_features_flag:
        features_to_vary = variable_features_names
    else:
        features_to_vary = 'all'

    print(datetime.datetime.now(), dataset, black_box, 'instability')

    for test_id, couple in enumerate(couples_to_test):
        print(datetime.datetime.now(), dataset, black_box, cfe, test_id, len(couples_to_test),
              '%.2f' % (test_id / len(couples_to_test)), 'instability')
        x1 = couple[0]
        x2 = couple[1]

        x1_dict = {k: v for k, v in zip(features_names, x1)}
        x2_dict = {k: v for k, v in zip(features_names, x2)}

        x_eval_list = list()
        for k in [1, 2, 3, 4, 5, 8, 10, 12, 14, 16, 18, 20]:
            time_start_i = datetime.datetime.now()

            dice_exp1 = exp.generate_counterfactuals(x1_dict,
                                                     total_CFs=k, desired_class='opposite',
                                                     features_to_vary=features_to_vary,
                                                     proximity_weight=0.5,
                                                     diversity_weight=1.0)
            cf_list1 = np.array(dice_exp1.final_cfs)
            cf_list1 = cf_list1.reshape((cf_list1.shape[0], cf_list1.shape[-1]))

            dice_exp2 = exp.generate_counterfactuals(x2_dict,
                                                     total_CFs=k, desired_class='opposite',
                                                     features_to_vary=features_to_vary,
                                                     proximity_weight=0.5,
                                                     diversity_weight=1.0)
            cf_list2 = np.array(dice_exp2.final_cfs)
            cf_list2 = cf_list2.reshape((cf_list2.shape[0], cf_list2.shape[-1]))

            time_test = (datetime.datetime.now() - time_start_i).total_seconds()

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
            x_eval['time_train'] = time_train
            x_eval['time_test'] = time_test
            x_eval['runtime'] = time_train + time_test
            x_eval['known_train'] = known_train
            x_eval['search_diversity'] = search_diversity
            x_eval['metric'] = metric if isinstance(metric, str) else '.'.join(metric)
            x_eval['variable_features_flag'] = variable_features_flag

            x_eval_list.append(x_eval)

        df = pd.DataFrame(data=x_eval_list)
        df = df[['dataset', 'black_box', 'method', 'couple_idx', 'k', 'known_train', 'search_diversity',
                 'metric', 'time_train', 'time_test', 'runtime', 'variable_features_flag', 'inst_x1x2']]

        if not os.path.isfile(filename_results):
            df.to_csv(filename_results, index=False)
        else:
            df.to_csv(filename_results, mode='a', index=False, header=False)

# NON FATTIBILE CAUSA TIPAGGIo DIFFERENTE DICE CON CONSTRAINTS E INSTABILITY
def main():

    nbr_test = 100
    dataset = 'compas'
    black_box = 'DNN'
    normalize = 'standard'
    variable_features_flag = False
    backend = 'TF2'

    # dataset = sys.argv[1]
    # black_box = sys.argv[2]
    # cfe = sys.argv[3]
    # nbr_test = 100 if len(sys.argv) < 5 else int(sys.argv[4])
    known_train = True
    search_diversity = False
    metric = 'none'
    nbr_exp = 20

    np.random.seed(random_state)

    if dataset not in dataset_list:
        print('unknown dataset %s' % dataset)
        return -1

    if black_box not in blackbox_list:
        print('unknown black box %s' % black_box)
        return -1

    print(datetime.datetime.now(), dataset, black_box)

    data = get_tabular_dataset(dataset, path_dataset, normalize=normalize, test_size=test_size,
                               random_state=random_state, return_original=True)
    df = data['df']
    class_name = data['class_name']
    class_values = data['class_values']
    if dataset == 'titanic':
        class_values = ['Not Survived', 'Survived']
    continuous_features_names = data['continuous_features_names']
    categorical_features_names = data['categorical_features_names']
    scaler = data['scaler']
    df[continuous_features_names] = scaler.transform(df[continuous_features_names])
    X_train_calc_dist = data['X_train']

    d = dice_ml.Data(dataframe=df, continuous_features=continuous_features_names, outcome_name=class_name)
    df_train, df_test = d.split_data(d.normalize_data(d.one_hot_encoded_data))
    X_train = df_train.loc[:, df_train.columns != class_name]
    y_train = df_train.loc[:, df_train.columns == class_name]
    X_test = df_test.loc[:, df_test.columns != class_name]
    y_test = df_test.loc[:, df_test.columns == class_name]
    # X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

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

    if black_box not in ['DNN']:
        print('black box %s not supported' % black_box)
        raise Exception

    bb_dice = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
    bb = BlackBox(bb_dice)

    metric = ('euclidean', 'jaccard')

    exp_calc_dist = RandomSACE(variable_features, weights=None, metric=metric, feature_names=None,
                               continuous_features=continuous_features,
                               categorical_features_lists=categorical_features_lists,
                               normalize=False, pooler=None, n_attempts=100, n_max_attempts=1000, proba=0.5)
    exp_calc_dist.fit(bb, X_train_calc_dist)

    y_pred = bb.predict(X_train_calc_dist)
    class_values = sorted(np.unique(y_pred))
    nbr_exp_per_class = nbr_exp // len(class_values)

    couples_to_test = list()
    for class_val in class_values:
        X_test_y = X_test[y_pred == class_val]
        for i, x in enumerate(X_test_y):
            neigh_dist = exp_calc_dist.cdist(x.reshape(1, -1), X_test_y)
            idx_neigh = np.argsort(neigh_dist, kind='stable')[0]
            closest_idx = idx_neigh[0] if i != idx_neigh[0] else idx_neigh[1]
            # print(class_val, i, closest_idx)
            couples_to_test.append((x, X_test_y[closest_idx]))
            if i >= nbr_exp_per_class:
                break

    filename_stability = path_results + 'instability_%s_%s_dice2.csv' % (dataset, black_box)
    metric = 'none'

    experiment('dice2', bb, bb_dice, d, X_train, variable_features, features_names, variable_features_names, X_test,
               nbr_test, dataset, black_box, continuous_features_all, categorical_features_all, ratio_cont,
               nbr_features, filename_stability, variable_features_flag, known_train, search_diversity,
               metric, backend, couples_to_test, exp_calc_dist)


if __name__ == "__main__":
    main()


