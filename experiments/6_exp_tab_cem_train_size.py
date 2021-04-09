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

from alibi.explainers import CEM

from cf_eval.metrics import *

from experiments.config import *
from experiments.util import get_tabular_dataset


def experiment(cfe, bb, X_train, variable_features, metric, continuous_features, categorical_features_lists,
               X_test, nbr_test, search_diversity, dataset, black_box, known_train, continuous_features_all,
               categorical_features_all, ratio_cont, nbr_features, filename_results, variable_features_flag,
               features_names, nbr_exp):

    for nbr_train_perc in np.arange(0.2, 1.2, 0.2):

        nbr_train = int(len(X_train) * nbr_train_perc)

        for id_exp in range(nbr_exp):

            X_train_sub_idx = np.random.choice(range(len(X_train)), nbr_train)
            X_train_sub = X_train[X_train_sub_idx]
            time_start = datetime.datetime.now()
            mode = 'PN'  # 'PN' (pertinent negative) or 'PP' (pertinent positive)
            shape = (1,) + X_train.shape[1:]  # instance shape
            kappa = .2  # minimum difference needed between the prediction probability for the perturbed instance on the
            # class predicted by the original instance and the max probability on the other classes
            # in order for the first loss term to be minimized
            beta = .1  # weight of the L1 loss term
            c_init = 10.  # initial weight c of the loss term encouraging to predict a different class (PN) or
            # the same class (PP) for the perturbed instance compared to the original instance to be explained
            c_steps = 10  # nb of updates for c
            max_iterations = 1000  # nb of iterations per value of c
            clip = (-1000., 1000.)  # gradient clipping
            lr_init = 1e-2  # initial learning rate

            predict_fn = lambda x: bb.predict_proba(x)  # only pass the predict fn which takes numpy arrays to CEM

            if not variable_features_flag:
                feature_range = (X_train_sub.min(axis=0).reshape(shape),  # feature range for the perturbed instance
                                 X_train_sub.max(axis=0).reshape(shape))  # can be either a float or array of shape (1xfeatures)

                # initialize CEM explainer and explain instance
                exp = CEM(predict_fn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range,
                          max_iterations=max_iterations, c_init=c_init, c_steps=c_steps,
                          learning_rate_init=lr_init, clip=clip)
                exp.fit(X_train_sub, no_info_type='median')  # we need to define what feature values contain the least
                # info wrt predictions
                # here we will naively assume that the feature-wise median
                # contains no info; domain knowledge helps!

            time_train = (datetime.datetime.now() - time_start).total_seconds()

            index_test_instances = np.random.choice(range(len(X_test)), nbr_test)

            print(datetime.datetime.now(), dataset, black_box, cfe, 'train size')

            for test_id, i in enumerate(index_test_instances):
                print(datetime.datetime.now(), dataset, black_box, cfe, test_id, len(index_test_instances),
                      '%.2f' % (test_id / len(index_test_instances)), 'train size')
                x = X_test[i]
                y_val = bb.predict(x.reshape(1, -1))[0]
                x_eval_list = list()
                cf_list_all = list()

                time_start_i = datetime.datetime.now()

                if variable_features_flag:
                    feature_range = (X_train_sub.min(axis=0).reshape(shape),  # feature range for the perturbed instance
                                     X_train_sub.max(axis=0).reshape(shape))  # can be either a float or array of shape (1xfeatures)

                    feature_range[0][:, variable_features] = x[variable_features]
                    feature_range[1][:, variable_features] = x[variable_features]

                    # initialize CEM explainer and explain instance
                    exp = CEM(predict_fn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range,
                              max_iterations=max_iterations, c_init=c_init, c_steps=c_steps,
                              learning_rate_init=lr_init, clip=clip)
                    exp.fit(X_train_sub, no_info_type='median')  # we need to define what feature values contain the least
                    # info wrt predictions
                    # here we will naively assume that the feature-wise median
                    # contains no info; domain knowledge helps!

                explanation = exp.explain(x.reshape(1, -1), verbose=False)
                cf_list = explanation.PN
                if cf_list is None:
                    cf_list = np.array([])

                time_test = (datetime.datetime.now() - time_start_i).total_seconds()

                for k in [
                    1,  # 2, 3, 4,
                    # 5,  # 8,
                    # 10,  # 12, 14,
                    # 15,  # 16, 18, 20
                ]:
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
                    x_eval['variable_features_flag'] = variable_features_flag

                    x_eval['id_exp_train_size'] = id_exp
                    x_eval['nbr_train_perc'] = nbr_train_perc
                    x_eval['nbr_train'] = nbr_train

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
                df = df[columns + ['id_exp_train_size', 'nbr_train_perc', 'nbr_train']]

                if not os.path.isfile(filename_results):
                    df.to_csv(filename_results, index=False)
                else:
                    df.to_csv(filename_results, mode='a', index=False, header=False)


def main():

    nbr_test = 20
    dataset = 'compas'
    black_box = 'RF'
    normalize = 'standard'

    # dataset = sys.argv[1]
    # black_box = sys.argv[2]
    # cfe = sys.argv[3]
    # nbr_test = 100 if len(sys.argv) < 5 else int(sys.argv[4])
    known_train = True
    search_diversity = False
    metric = 'none'
    variable_features_flag = True
    nbr_exp_per_train_size = 5

    np.random.seed(random_state)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.compat.v1.disable_eager_execution()

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

    filename_train_size = path_results + 'train_size_%s_%s_cem.csv' % (dataset, black_box)

    experiment('cem', bb, X_train, variable_features, metric,
               continuous_features, categorical_features_lists,
               X_test, nbr_test, search_diversity, dataset, black_box, known_train,
               continuous_features_all, categorical_features_all, ratio_cont, nbr_features,
               filename_train_size, variable_features_flag, features_names, nbr_exp_per_train_size)


if __name__ == "__main__":
    main()


