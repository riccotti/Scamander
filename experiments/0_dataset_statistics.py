from experiments.util import *
from experiments.config import path_dataset


def main():

    print('dataset, rows, cols, cont, cate, classes, cols1h, var, cont var, cate var, cate var 1h')
    for dataset in ['adult', 'bank', 'churn', 'compas', 'diabetes', 'fico', 'german', 'home', 'titanic']:
        # df, class_name = get_tabular_dataset(dataset, path_dataset)
        # df = remove_missing_values(df)
        # features = [c for c in df.columns if c not in [class_name]]
        # continuous_features_names = list(df[features]._get_numeric_data().columns)
        # nbr_rows = len(df)
        # nbr_cols = len(features)
        # nbr_cont_cols = len(continuous_features_names)
        # nbr_cat_cols = nbr_cols - nbr_cont_cols
        # nbr_classes = len(df[class_name].unique())
        #
        # df, feature_names, class_values = one_hot_encoding(df, class_name)
        # categorical_features_lists_all = get_categorical_features_lists(feature_names)
        # variable_features_names = dataset_variable_features_names[dataset]
        # variable_features = [i for i, f in enumerate(feature_names) if f in variable_features_names]
        #
        # continuous_features = list()
        # for i, f in enumerate(feature_names):
        #     if f in continuous_features_names and f in variable_features_names:
        #         continuous_features.append(i)
        #
        # categorical_features_lists = list()
        # for index in categorical_features_lists_all:
        #     if index[0] in variable_features:
        #         categorical_features_lists.append(index)
        #
        # nbr_cols1h = len(feature_names)
        # nbr_var = len(variable_features)
        # nbr_cont_var = len(continuous_features)
        # nbr_cate_var = len(categorical_features_lists)
        # nbr_cate_var1h = nbr_var - nbr_cont_var
        # print(class_name, '<----')

        # print('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s' % (
        #     dataset, nbr_rows, nbr_cols, nbr_cont_cols, nbr_cat_cols, nbr_classes,
        #     nbr_cols1h, nbr_var, nbr_cont_var, nbr_cate_var, nbr_cate_var1h))

        # if dataset in ['adult', 'bank', 'churn', 'compas', 'diabetes', 'fico', 'german', 'home']:
        #     continue
        # for c in feature_names:
        #     if c == class_name:
        #         continue
        #     print("'%s'," % c)
        # break

        data = get_tabular_dataset(dataset, path_dataset)

        print('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s' % (
            dataset, data['n_rows'], data['n_cols'], data['n_cont_cols'], data['n_cate_cols'], data['n_classes'],
            data['n_cols1h'], data['n_var'], data['n_cont_var'], data['n_cate_var'], data['n_cate_var1h']))


if __name__ == "__main__":
    main()
