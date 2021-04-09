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


def experiment(cfe, bb, bb_dice, X_train, variable_features, features_names, variable_cont_features_names,
               variable_cate_features_names, X_test, nbr_test, dataset, black_box, continuous_features_all,
               categorical_features_all, ratio_cont, nbr_features, filename_results,
               variable_features_flag, known_train, search_diversity, metric, backend):

    time_start = datetime.datetime.now()
    df = pd.DataFrame(data=X_train, columns=features_names)
    d = dice_ml.Data(dataframe=df, continuous_features=features_names, outcome_name='class')
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
        features_to_vary = variable_cont_features_names + variable_cate_features_names
    else:
        features_to_vary = 'all'

    index_test_instances = np.random.choice(range(len(X_test)), nbr_test)

    print(datetime.datetime.now(), dataset, black_box, 'plausibility')

    for test_id, i in enumerate(index_test_instances):
        print(datetime.datetime.now(), dataset, black_box, cfe, test_id, len(index_test_instances),
              '%.2f' % (test_id / len(index_test_instances)), 'plausibility')
        x = X_test[i]
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

            x_eval = evaluate_only_plasubility(cf_list, x, bb, y_val, k, variable_features, continuous_features_all,
                                               categorical_features_all, X_train, X_test, ratio_cont)

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

        df = pd.DataFrame(data=x_eval_list)
        df = df[['dataset', 'black_box', 'method', 'idx', 'k', 'known_train', 'search_diversity',
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

    nbr_test = 20
    dataset = 'compas'
    black_box = 'DNN'
    normalize = 'standard'
    variable_features_flag = False
    # dataset = sys.argv[1]
    # black_box = sys.argv[2]
    # cfe = sys.argv[3]
    # nbr_test = 100 if len(sys.argv) < 5 else int(sys.argv[4])
    known_train = True
    search_diversity = False
    metric = 'none'
    backend = 'TF2'

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

    if black_box not in ['DNN']:
        print('black box %s not supported' % black_box)
        raise Exception

    bb_dice = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
    bb = BlackBox(bb_dice)

    filename_plusibility = path_results + 'plausibility_%s_%s_dice.csv' % (dataset, black_box)

    experiment('dice', bb, bb_dice, X_train, variable_features, features_names, variable_cont_features_names,
               variable_cate_features_names, X_test, nbr_test, dataset, black_box,
               continuous_features_all, categorical_features_all, ratio_cont, nbr_features,
               filename_plusibility, variable_features_flag, known_train, search_diversity, metric,
               backend)


if __name__ == "__main__":
    main()


