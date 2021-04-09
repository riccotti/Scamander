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


def experiment(cfe, bb, bb_dice, d, X_train, variable_features, features_names, variable_features_names,
               X_test, nbr_test, dataset, black_box, continuous_features_all,
               categorical_features_all, ratio_cont, nbr_features, filename_results,
               variable_features_flag, known_train, search_diversity, metric, backend, filename_cf):

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

    index_test_instances = np.random.choice(range(len(X_test)), nbr_test)

    print(datetime.datetime.now(), dataset, black_box)

    for test_id, i in enumerate(index_test_instances):
        print(datetime.datetime.now(), dataset, black_box, cfe, test_id, len(index_test_instances),
              '%.2f' % (test_id / len(index_test_instances)))
        x = X_test.values[i]
        x_dict = {k: v for k, v in zip(features_names, x)}
        y_val = bb.predict(x.reshape(1, -1))[0]

        cf_list_all = list()
        x_eval_list = list()
        for k in [1, 2, 3, 4, 5, 8, 10, 12, 14, 16, 18, 20]:
            time_start_i = datetime.datetime.now()

            dice_exp = exp.generate_counterfactuals(x_dict,
                                                    total_CFs=k, desired_class='opposite',
                                                    features_to_vary=features_to_vary,
                                                    proximity_weight=0.5,
                                                    diversity_weight=1.0)
            cf_list = np.array(dice_exp.final_cfs)
            cf_list = cf_list.reshape((cf_list.shape[0], cf_list.shape[-1]))

            time_test = (datetime.datetime.now() - time_start_i).total_seconds()

            x_eval = evaluate_cf_list(cf_list, x, bb, y_val, k, variable_features,
                                      continuous_features_all, categorical_features_all, X_train.values, X_test.values,
                                      ratio_cont, nbr_features)

            x_eval['dataset'] = dataset
            x_eval['black_box'] = black_box
            x_eval['method'] = cfe
            x_eval['idx'] = i
            x_eval['k'] = k
            x_eval['time_train'] = time_train
            x_eval['time_test'] = time_test
            x_eval['runtime'] = time_train + time_test
            x_eval['known_train'] = known_train
            x_eval['search_diversity'] = search_diversity
            x_eval['metric'] = metric if isinstance(metric, str) else '.'.join(metric)
            x_eval['variable_features_flag'] = variable_features_flag

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
    dataset = 'adult'
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

    filename_results = path_results + 'cfeval_%s_%s_dice2.csv' % (dataset, black_box)
    filename_cf = path_cf + 'cf_%s_%s_dice2.csv' % (dataset, black_box)

    experiment('dice2', bb, bb_dice, d, X_train, variable_features, features_names, variable_features_names, X_test,
               nbr_test, dataset, black_box, continuous_features_all, categorical_features_all, ratio_cont,
               nbr_features, filename_results, variable_features_flag, known_train, search_diversity,
               metric, backend, filename_cf)


if __name__ == "__main__":
    main()


