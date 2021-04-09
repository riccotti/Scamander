import os
import pickle
import imageio
import numpy as np
import pandas as pd

from os import listdir

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.datasets import make_moons, make_circles, make_classification

from tslearn.datasets import UCR_UEA_datasets
from pyts.preprocessing import MinMaxScaler as TsMinMaxScaler, StandardScaler as TsStandardScaler

from skimage import filters
from sklearn.datasets import make_classification

from keras.datasets import mnist, cifar10, cifar100, fashion_mnist

from tensorflow.keras.preprocessing import image


datasets = {
    'avila': ('avila.csv', 'tab'),
    'adult': ('adult.csv', 'tab'),
    'bank': ('bank.csv', 'tab'),
    'churn': ('churn.csv', 'tab'),
    'ctg': ('ctg.csv', 'tab'),
    'compas': ('compas-scores-two-years.csv', 'tab'),
    'fico': ('fico.csv', 'tab'),
    'diabetes': ('diabetes.csv', 'tab'),
    'german': ('german_credit.csv', 'tab'),
    'home': ('home.csv', 'tab'),
    'ionoshpere': ('ionosphere.csv', 'tab'),
    'mouse': ('mouse.csv', 'tab'),
    'parkinsons': ('parkinsons.csv', 'tab'),
    'sonar': ('sonar.csv', 'tab'),
    'vehicle': ('vehicle.csv', 'tab'),
    'wdbc': ('wdbc.csv', 'tab'),
    'wine': ('wine.csv', 'tab'),
    'winer': ('wine-red.csv', 'tab'),
    'winew': ('wine-white.csv', 'tab'),
    'titanic': ('titanic.csv', 'tab'),
    'iris': ('iris.csv', 'tab'),
    'rnd': ('', 'rnd'),

    'moons': ('', 'moons'),
    'circles': ('', 'circles'),
    'linear': ('', 'linear'),

    'mnist': ('', 'img'),
    'fashion_mnist': ('', 'img'),
    'cifar10': ('', 'img'),
    'cifar100': ('', 'img'),
    'omniglot': ('omniglot', 'img'),
    'imagenet1000': ('imagenet/', 'img'),

    'gunpoint': ('GunPoint', 'ts'),
    'italypower': ('ItalyPowerDemand', 'ts'),
    'arrowhead': ('ArrowHead', 'ts'),
    'ecg200': ('ECG200', 'ts'),
    'ecg5000': ('ECG5000', 'ts'),
    'electricdevices': ('ElectricDevices', 'ts'),
    'phalanges': ('PhalangesOutlinesCorrect', 'ts'),
    'diatom': ('DiatomSizeReduction', 'ts'),
    'ecg5days': ('ECGFiveDays', 'ts'),
    'facefour': ('FaceFour', 'ts'),
    'herring': ('Herring', 'ts'),

    '20newsgroups': ('', 'txt'),
    'imdb': ('', 'txt'),
}


def remove_missing_values(df):
    for column_name, nbr_missing in df.isna().sum().to_dict().items():
        if nbr_missing > 0:
            if column_name in df._get_numeric_data().columns:
                mean = df[column_name].mean()
                df[column_name].fillna(mean, inplace=True)
            else:
                mode = df[column_name].mode().values[0]
                df[column_name].fillna(mode, inplace=True)
    return df


def one_hot_encoding(df, class_name):
    """
    Return one hot encoding of the dataframe passed
    :param df: input dataframe
    :param class_name: target class name
    :return: one hot encoded dataframe
    """

    dfX = pd.get_dummies(df[[c for c in df.columns if c not in [class_name]]], prefix_sep='=')
    class_name_map = {v: k for k, v in enumerate(sorted(df[class_name].unique()))}
    dfY = df[class_name].map(class_name_map)
    df = pd.concat([dfX, dfY], axis=1) #, join_axes=[dfX.index])
    feature_names = list(dfX.columns)
    class_values = sorted(class_name_map)

    return df, feature_names, class_values


def get_categorical_features_lists(feature_names):
    i = 0
    categorical_features_names = list()
    categorical_features_lists = list()
    while i < len(feature_names) - 1:
        if '=' not in feature_names[i]:
            i += 1
            continue
        fn0, val0 = feature_names[i].split('=')[:2]
        values = [val0]
        indexes = [i]
        for j in range(i+1, len(feature_names)):
            fn1, val1 = feature_names[j].split('=')[:2]
            i = j
            if fn1 == fn0:
                values.append(val1)
                indexes.append(j)
                # i = j
            else:
                # i = j
                break
            # i = j
        categorical_features_names.append([fn0, values])
        categorical_features_lists.append(indexes)
        # i = j
    return categorical_features_lists


def get_titanic_dataset(filename):
    class_name = 'Survived'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    df['Family'] = df['SibSp'] + df['Parch']
    df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'SibSp', 'Parch'], axis=1, inplace=True)
    df['Age'] = df['Age'].fillna(np.mean(df['Age']))
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df, class_name


def get_iris_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    return df, class_name


def get_moons_dataset(filename):
    class_name = 'class'
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=0)
    data = np.hstack([X, y.reshape(-1, 1)])
    columns = ['x0', 'x1', class_name]
    df = pd.DataFrame(data=data, columns=columns)
    return df, class_name


def get_circles_dataset(filename):
    class_name = 'class'
    X, y = make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=1)
    data = np.hstack([X, y.reshape(-1, 1)])
    columns = ['x0', 'x1', class_name]
    df = pd.DataFrame(data=data, columns=columns)
    return df, class_name


def get_linear_dataset(filename):
    class_name = 'class'
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    data = np.hstack([X, y.reshape(-1, 1)])
    columns = ['x0', 'x1', class_name]
    df = pd.DataFrame(data=data, columns=columns)
    return df, class_name


def get_compas_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    columns = ['age',  # 'age_cat',
               'sex', 'race', 'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']

    df = df[columns]
    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])
    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)
    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)
    df['class'] = df['score_text']
    df.drop(['c_jail_in', 'c_jail_out', 'decile_score', 'score_text'], axis=1, inplace=True)

    return df, class_name


def get_adult_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True, na_values='?', keep_default_na=True)
    df.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)
    return df, class_name


def get_home_dataset(filename):
    class_name = 'in_sf'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True, na_values='?', keep_default_na=True)
    return df, class_name


def get_german_dataset(filename):
    class_name = 'default'
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = [c.replace('=', '') for c in df.columns]
    return df, class_name


def get_fico_dataset(filename):
    class_name = 'RiskPerformance'
    df = pd.read_csv(filename, skipinitialspace=True, keep_default_na=True)
    return df, class_name


def get_churn_dataset(filename):
    class_name = 'churn'
    df = pd.read_csv(filename, skipinitialspace=True, na_values='?', keep_default_na=True)
    columns2remove = ['phone number']
    df.drop(columns2remove, inplace=True, axis=1)
    return df, class_name


def get_bank_dataset(filename):
    class_name = 'give_credit'
    df = pd.read_csv(filename, skipinitialspace=True, keep_default_na=True, index_col=0)
    return df, class_name


def get_avila_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    df = df[df['class'] != 'B']
    df = df[df['class'] != 'W']
    return df, class_name


def get_ctg_dataset(filename):
    class_name = 'CLASS'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    df.drop(['FileName', 'Date', 'SegFile', 'NSP'], axis=1, inplace=True)
    return df, class_name


def get_diabetes_dataset(filename):
    class_name = 'Outcome'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    return df, class_name


def get_ionosphere_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    return df, class_name


def get_mouse_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    df.drop(['MouseID', 'Genotype', 'Treatment', 'Behavior'], axis=1, inplace=True)
    return df, class_name


def get_parkinsons_dataset(filename):
    class_name = 'status'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    df.drop(['name'], axis=1, inplace=True)
    return df, class_name


def get_wdbc_dataset(filename):
    class_name = 'diagnosis'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    df.drop(['id'], axis=1, inplace=True)
    return df, class_name


def get_sonar_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    return df, class_name


def get_vehicle_dataset(filename):
    class_name = 'CLASS'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    return df, class_name


def get_wine_dataset(filename):
    class_name = 'quality'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    return df, class_name


def get_arrowhead_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('ArrowHead')
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_gunpoint_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('GunPoint')
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_ecg200_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('ECG200')
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_ecg5000_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('ECG5000')
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_italypower_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('ItalyPowerDemand')
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [2, 4, 6]
    window_steps = [1, 1, 1]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_electricdevices_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('ElectricDevices')
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_phalanges_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('PhalangesOutlinesCorrect')
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_diatom_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('DiatomSizeReduction')
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_ecg5days_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('ECGFiveDays')
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_facefour_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('FaceFour')
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_herring_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('Herring')
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_random_dataset(n_samples, n_features, n_informative, n_classes, test_size, random_state):
    class_name = 'class'
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_classes=n_classes,
                               random_state=random_state,
                               shuffle=False)
    columns = [str(c) for c in range(n_features)] + [class_name]
    df = pd.DataFrame(data=np.concatenate([X, y.reshape(len(y), 1)], axis=1), columns=columns)

    feature_names = [c for c in df.columns if c != class_name]
    class_values = sorted(np.unique(df[class_name]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

    dataset = {
        'name': 'rnd',
        'data_type': 'rnd',
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'class_name': class_name,
        'class_values': class_values,
        'feature_names': feature_names,
        'n_features': n_features,
        'n_classes': n_classes,
        'n_samples': n_samples,
        'n_informative': n_informative
    }

    return dataset


def get_mnist_dataset(filename=None, categories=None):
    w, h = 28, 28
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], w, h)
    X_test = X_test.reshape(X_test.shape[0], w, h)

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    if categories is not None:
        idx_train = np.array([np.where(c == y_train)[0] for c in categories])
        idx_train = np.concatenate(idx_train)
        idx_test = np.array([np.where(c == y_test)[0] for c in categories])
        idx_test = np.concatenate(idx_test)
        X_train, y_train = X_train[idx_train], y_train[idx_train]
        X_test, y_test = X_test[idx_test], y_test[idx_test]

    window_sizes = [(7, 7), (14, 14), (14, 14)]
    window_steps = [(7, 7), (7, 7), (14, 14)]

    return X_train, X_test, y_train, y_test, w, h, window_sizes, window_steps


def get_fashion_mnist_dataset(filename=None, categories=None):
    w, h = 28, 28
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], w, h)
    X_test = X_test.reshape(X_test.shape[0], w, h)

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    if categories is not None:
        idx_train = np.array([np.where(c == y_train)[0] for c in categories])
        idx_train = np.concatenate(idx_train)
        idx_test = np.array([np.where(c == y_test)[0] for c in categories])
        idx_test = np.concatenate(idx_test)
        X_train, y_train = X_train[idx_train], y_train[idx_train]
        X_test, y_test = X_test[idx_test], y_test[idx_test]

    window_sizes = [(7, 7), (14, 14), (14, 14)]
    window_steps = [(7, 7), (7, 7), (14, 14)]

    return X_train, X_test, y_train, y_test, w, h, window_sizes, window_steps


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def get_cifar10_dataset(filename=None, categories=None):
    w, h = 32, 32
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # X_train = np.array([rgb2gray(x) for x in X_train])
    # X_test = np.array([rgb2gray(x) for x in X_test])

    X_train = X_train.reshape((X_train.shape[0], w, h, 3))
    X_test = X_test.reshape((X_test.shape[0], w, h, 3))

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    if categories is not None:
        idx_train = np.array([np.where(c == y_train)[0] for c in categories])
        idx_train = np.concatenate(idx_train)
        idx_test = np.array([np.where(c == y_test)[0] for c in categories])
        idx_test = np.concatenate(idx_test)
        X_train, y_train = X_train[idx_train], y_train[idx_train]
        X_test, y_test = X_test[idx_test], y_test[idx_test]

    window_sizes = [(8, 8), (16, 16), (16, 16)]
    window_steps = [(8, 8), (8, 8), (16, 16)]

    return X_train, X_test, y_train, y_test, w, h, window_sizes, window_steps


def get_cifar100_dataset(filename=None, categories=None):
    w, h = 32, 32
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    # X_train = np.array([rgb2gray(x) for x in X_train])
    # X_test = np.array([rgb2gray(x) for x in X_test])

    X_train = X_train.reshape((X_train.shape[0], w, h, 3))
    X_test = X_test.reshape((X_test.shape[0], w, h, 3))

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    if categories is not None:
        idx_train = np.where(categories == y_train)[0]
        idx_test = np.where(categories == y_test)[0]
        X_train, y_train = X_train[idx_train], y_train[idx_train]
        X_test, y_test = X_test[idx_test], y_test[idx_test]

    window_sizes = [(8, 8), (16, 16), (16, 16)]
    window_steps = [(8, 8), (8, 8), (16, 16)]

    return X_train, X_test, y_train, y_test, w, h, window_sizes, window_steps


def get_imagenet1000_dataset(filename=None, categories=None):
    w, h = 224, 224

    imagefiles = [f for f in listdir(filename) if f.endswith('JPEG')]
    labelsfile = filename + 'imagenet_labels.csv'
    df = pd.read_csv(labelsfile)
    df.set_index('filename', inplace=True)
    # filename_class_map = df[['class_value']].to_dict()['class_value']
    filename_class_map = df[['class']].to_dict()['class']
    # print(filename_class_map)
    X = list()
    y = list()
    for f in imagefiles:
        img = image.load_img(filename + f, target_size=(w, h))
        x = image.img_to_array(img)
        X.append(x)
        y.append(filename_class_map[f])
    X = np.array(X)
    y = np.array(y)

    if categories is not None:
        idx = np.where(categories == y)[0]
        X, y = X[idx], y[idx]

    window_sizes = [(8, 8), (16, 16), (16, 16)]
    window_steps = [(8, 8), (8, 8), (16, 16)]

    return None, X, None, y, w, h, window_sizes, window_steps


dataset_read_function_map = {
    # '20newsgroups': get_20news_dataset,
    # 'imdb': get_imdb_dataset,

    'arrowhead': get_arrowhead_dataset,
    'ecg200': get_ecg200_dataset,
    'ecg5000': get_ecg5000_dataset,
    'diatom': get_diatom_dataset,
    'ecg5days': get_ecg5days_dataset,
    'facefour': get_facefour_dataset,
    'herring': get_herring_dataset,
    'gunpoint': get_gunpoint_dataset,
    'italypower': get_italypower_dataset,
    'electricdevices': get_electricdevices_dataset,
    'phalanges': get_phalanges_dataset,

    'avila': get_avila_dataset,
    'ctg': get_ctg_dataset,
    'diabetes': get_diabetes_dataset,
    'mouse': get_mouse_dataset,
    'ionoshpere': get_ionosphere_dataset,
    'parkinsons': get_parkinsons_dataset,
    'sonar': get_sonar_dataset,
    'vehicle': get_vehicle_dataset,
    'wdbc': get_wdbc_dataset,
    'wine': get_wine_dataset,
    'winer': get_wine_dataset,
    'winew': get_wine_dataset,

    'adult': get_adult_dataset,
    'bank': get_bank_dataset,
    'churn': get_churn_dataset,
    'compas': get_compas_dataset,
    'fico': get_fico_dataset,
    'german': get_german_dataset,
    'home': get_home_dataset,
    'titanic': get_titanic_dataset,
    'iris': get_iris_dataset,

    'moons': get_moons_dataset,
    'circles': get_circles_dataset,
    'linear': get_linear_dataset,

    'rnd': get_random_dataset,

    'cifar10': get_cifar10_dataset,
    'cifar100': get_cifar100_dataset,
    'mnist': get_mnist_dataset,
    'fashion_mnist': get_fashion_mnist_dataset,
    'imagenet1000': get_imagenet1000_dataset,
}


def get_tabular_dataset(name, path='./', normalize=None, test_size=0.3, random_state=None, return_original=False,
                        encode='onehot'):

    get_dataset_fn = dataset_read_function_map[name]

    filename = path + datasets[name][0]
    data_type = datasets[name][1]

    df, class_name = get_dataset_fn(filename)

    if return_original:
        dfo = df.copy()
    else:
        dfo = None

    df = remove_missing_values(df)
    features = [c for c in df.columns if c not in [class_name]]
    continuous_features_names = list(df[features]._get_numeric_data().columns)
    categorical_features_names = [c for c in df.columns if c not in continuous_features_names and c != class_name]
    n_rows = len(df)
    n_cols = len(features)
    n_cont_cols = len(continuous_features_names)
    n_cate_cols = n_cols - n_cont_cols
    n_classes = len(df[class_name].unique())

    if encode == 'onehot':
        df, feature_names, class_values = one_hot_encoding(df, class_name)
        categorical_features_lists_all = get_categorical_features_lists(feature_names)
        variable_features_names = dataset_variable_features_names[name]
        variable_features = [i for i, f in enumerate(feature_names) if f in variable_features_names]
    elif encode in ['none', None]:
        feature_names = continuous_features_names + categorical_features_names
        df = df[feature_names + [class_name]]
        class_name_map = {v: k for k, v in enumerate(sorted(df[class_name].unique()))}
        class_values = sorted(class_name_map)

        categorical_features_lists_all = [[cidx] for cidx in np.arange(len(continuous_features_names),
                                                                       len(feature_names), 1)]
        variable_features_names = dataset_variable_features_names_noencode[name]
        variable_features = [i for i, f in enumerate(feature_names) if f in variable_features_names]
    else:
        raise Exception('Unknown encoding %s' % encode)

    continuous_features = list()
    continuous_features_all = list()
    for i, f in enumerate(feature_names):
        if f in continuous_features_names:
            continuous_features_all.append(i)
            if f in variable_features_names:
                continuous_features.append(i)

    categorical_features = list()
    categorical_features_lists = list()
    categorical_features_all = list()
    for index in categorical_features_lists_all:
        categorical_features_all.extend(index)
        if index[0] in variable_features:
            categorical_features.extend(index)
            categorical_features_lists.append(index)

    n_cols1h = len(feature_names)
    n_var = len(variable_features)
    n_cont_var = len(continuous_features)
    n_cate_var = len(categorical_features_lists)
    n_cate_var1h = n_var - n_cont_var

    X = df[feature_names].values
    y = df[class_name].values

    if normalize == 'minmax':
        # scaler = MinMaxScaler()
        # X = scaler.fit_transform(X)
        # scaler = ColumnTransformer(transformers=[(normalize, StandardScaler(), continuous_features_all)],
        #                            remainder='passthrough')
        # X = scaler.fit_transform(X)
        scaler = MinMaxScaler()
        X[:, continuous_features_all] = scaler.fit_transform(X[:, continuous_features_all])
    elif normalize == 'standard':
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)
        # scaler = ColumnTransformer(transformers=[(normalize, StandardScaler(), continuous_features_all)],
        #                            remainder='passthrough')
        # X = scaler.fit_transform(X)
        scaler = StandardScaler()
        X[:, continuous_features_all] = scaler.fit_transform(X[:, continuous_features_all])
    else:
        scaler = None

    if encode in ['none', None]:
        encoder = OrdinalEncoder()
        encoder_y = LabelEncoder()
        if len(categorical_features_all):
            X_cat = encoder.fit_transform(X[:, categorical_features_all])
            X_con = X[:, continuous_features_all]
            X = np.hstack([X_con, X_cat]).astype(np.float64)
        y = encoder_y.fit_transform(y)
    else:
        encoder = None
        encoder_y = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

    data = {
        'name': name,
        'data_type': data_type,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'class_name': class_name,
        'class_values': class_values,
        'feature_names': feature_names,
        'continuous_features_names': continuous_features_names,
        'categorical_features_names': categorical_features_names,

        'n_classes': n_classes,
        'n_rows': n_rows,
        'n_cols': n_cols,
        'n_cont_cols': n_cont_cols,
        'n_cate_cols': n_cate_cols,
        'n_cols1h': n_cols1h,
        'n_var': n_var,
        'n_cont_var': n_cont_var,
        'n_cate_var': n_cate_var,
        'n_cate_var1h': n_cate_var1h,

        'variable_features': variable_features,
        'variable_features_names': variable_features_names,
        'continuous_features': continuous_features,
        'continuous_features_all': continuous_features_all,
        'categorical_features': categorical_features,
        'categorical_features_all': categorical_features_all,
        'categorical_features_lists': categorical_features_lists,
        'categorical_features_lists_all': categorical_features_lists_all,

        'scaler': scaler,
        'df': dfo,
        'encoder': encoder,
        'encoder_y': encoder_y
    }

    return data


def get_image_dataset(name, path, categories, filter, use_rgb=True, flatten=False, model=None, expand_dims=False):

    get_dataset_fn = dataset_read_function_map[name]

    filename = path + datasets[name][0]
    data_type = datasets[name][1]

    X_train, X_test, y_train, y_test, w, h, window_sizes, window_steps = get_dataset_fn(filename, categories)

    if not use_rgb:
        X_train = np.array([rgb2gray(x) for x in X_train])
        X_test = np.array([rgb2gray(x) for x in X_test])

    if filter == 'sobel':
        X_train = np.array([filters.sobel(x) for x in X_train])
        X_test = np.array([filters.sobel(x) for x in X_test])
    elif filter == 'roberts':
        X_train = np.array([filters.roberts(x) for x in X_train])
        X_test = np.array([filters.roberts(x) for x in X_test])

    if model is not None:
        if model == 'InceptionV3':
            from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
        elif model == 'VGG16':
            from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
        elif model == 'VGG19':
            from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
        elif model == 'ResNet50':
            from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
        elif model == 'InceptionResNetV2':
            from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
        else:
            raise Exception('unkown model %s' % model)

        X_test_new = np.zeros(X_test.shape)
        for i, x in enumerate(X_test):
            X_test_new[i] = preprocess_input(np.expand_dims(x, axis=0))

    # if X_train is not None and len(X_train.shape) < 4 and model is not None:
    if expand_dims:
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)

    if flatten:
        X_train = np.array([x.ravel() for x in X_train])
        X_test = np.array([x.ravel() for x in X_test])

    class_name = 'class'
    class_values = sorted(np.unique(y_train))
    n_classes = len(class_values)

    dataset = {
        'name': name,
        'data_type': data_type,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'class_name': class_name,
        'class_values': class_values,
        'word_size': w,
        'h': h,
        'n_classes': n_classes,
        'window_sizes': window_sizes,
        'window_steps': window_steps,
    }

    if model is not None:
        dataset['decode_predictions'] = decode_predictions

    return dataset


def get_ts_dataset(name, path, normalize=None):
    get_dataset_fn = dataset_read_function_map[name]

    filename = path + datasets[name][0]
    data_type = datasets[name][1]

    X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps = get_dataset_fn(filename)

    if normalize is not None:
        shape_train = X_train.shape
        shape_test = X_test.shape
        X_train = X_train.reshape(shape_train[:-1])
        X_test = X_test.reshape(shape_test[:-1])

        if normalize == 'minmax':
            scaler = TsMinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif normalize in ['standard', 'normal']:
            scaler = TsStandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        X_train = X_train.reshape(shape_train)
        X_test = X_test.reshape(shape_test)

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    class_name = 'class'
    class_values = sorted(np.unique(y_train))
    n_classes = len(class_values)

    dataset = {
        'name': name,
        'data_type': data_type,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'class_name': class_name,
        'class_values': class_values,
        'n_timestamps': n_timestamps,
        'n_classes': n_classes,
        'window_sizes': window_sizes,
        'window_steps': window_steps
    }

    return dataset


def get_dataset(name, path='./', normalize=None, test_size=0.3, random_state=None, **kwargs):

    if name not in datasets:
        raise ValueError('Unknown dataset %s' % name)

    dataset_type = datasets[name][1]

    if dataset_type == 'tab':
        return get_tabular_dataset(name, path, normalize, test_size, random_state)
    elif dataset_type == 'img':
        categories = kwargs.get('categories', None)
        filter = kwargs.get('filter', None)
        use_rgb = kwargs.get('use_rgb', False)
        flatten = kwargs.get('flatten', False)
        return get_image_dataset(name, path, categories, filter, use_rgb, flatten)
    elif dataset_type == 'ts':
        return get_ts_dataset(name, path)
    # elif dataset_type == 'txt':
    #     n_words = kwargs.get('n_words', 1000)
    #     encoding = kwargs.get('encoding', 'tfidf')
    #     categories = kwargs.get('categories', None)
    #     return get_txt_dataset(name, path, n_words, encoding, categories)
    elif dataset_type == 'rnd':
        min_samples = kwargs.get('min_samples', 10000)
        max_samples = kwargs.get('max_samples', 100000)
        min_features = kwargs.get('min_features', 10)
        max_features = kwargs.get('max_features', 1000)
        min_classes = kwargs.get('min_classes', 2)
        max_classes = kwargs.get('max_classes', 20)
        n_samples = np.random.randint(min_samples, max_samples + 1)
        n_features = np.random.randint(min_features, max_features + 1)
        n_informative = np.random.randint(0, n_features // 2)
        n_classes = np.random.randint(min_classes, max_classes + 1)
        return get_random_dataset(n_samples, n_features, n_informative, n_classes, test_size, random_state)



dataset_variable_features_names = {
    'adult': [
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'workclass=Federal-gov',
        'workclass=Local-gov',
        'workclass=Never-worked',
        'workclass=Private',
        'workclass=Self-emp-inc',
        'workclass=Self-emp-not-inc',
        'workclass=State-gov',
        'workclass=Without-pay',
        'occupation=Adm-clerical',
        'occupation=Armed-Forces',
        'occupation=Craft-repair',
        'occupation=Exec-managerial',
        'occupation=Farming-fishing',
        'occupation=Handlers-cleaners',
        'occupation=Machine-op-inspct',
        'occupation=Other-service',
        'occupation=Priv-house-serv',
        'occupation=Prof-specialty',
        'occupation=Protective-serv',
        'occupation=Sales',
        'occupation=Tech-support',
        'occupation=Transport-moving',
    ],
    'bank': [
        'revolving',
        'nbr_30_59_days_past_due_not_worse',
        'debt_ratio',
        'monthly_income',
        'nbr_open_credits_and_loans',
        'nbr_90_days_late',
        'nbr_real_estate_loans_or_lines',
        'nbr_60_89_days_past_due_not_worse',
        'dependents',
    ],
    'churn': [
        'account length',
        'number vmail messages',
        'total day minutes',
        'total day calls',
        'total day charge',
        'total eve minutes',
        'total eve calls',
        'total eve charge',
        'total night minutes',
        'total night calls',
        'total night charge',
        'total intl minutes',
        'total intl calls',
        'total intl charge',
        'customer service calls',
        'international plan=no',
        'international plan=yes',
        'voice mail plan=no',
        'voice mail plan=yes',
    ],
    'compas': [
        'priors_count',
        'days_b_screening_arrest',
        'is_recid',
        'is_violent_recid',
        'two_year_recid',
        'length_of_stay',
        'c_charge_degree=F',
        'c_charge_degree=M',
    ],
    'diabetes': [
        # 'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        # 'DiabetesPedigreeFunction',
        # 'Age',
    ],
    'fico': [
        # 'ExternalRiskEstimate',
        'MSinceOldestTradeOpen',
        'MSinceMostRecentTradeOpen',
        'AverageMInFile',
        'NumSatisfactoryTrades',
        'NumTrades60Ever2DerogPubRec',
        'NumTrades90Ever2DerogPubRec',
        'PercentTradesNeverDelq',
        'MSinceMostRecentDelq',
        'MaxDelq2PublicRecLast12M',
        'MaxDelqEver',
        'NumTotalTrades',
        'NumTradesOpeninLast12M',
        'PercentInstallTrades',
        'MSinceMostRecentInqexcl7days',
        'NumInqLast6M',
        'NumInqLast6Mexcl7days',
        'NetFractionRevolvingBurden',
        'NetFractionInstallBurden',
        'NumRevolvingTradesWBalance',
        'NumInstallTradesWBalance',
        'NumBank2NatlTradesWHighUtilization',
        'PercentTradesWBalance',
    ],
    'german': [
        'duration_in_month',
        'credit_amount',
        'installment_as_income_perc',
        'present_res_since',
        # 'age',
        'credits_this_bank',
        # 'people_under_maintenance',
        'account_check_status=0 <= ... < 200 DM',
        'account_check_status=< 0 DM',
        'account_check_status=>= 200 DM / salary assignments for at least 1 year',
        'account_check_status=no checking account',
        # 'credit_history=all credits at this bank paid back duly',
        # 'credit_history=critical account/ other credits existing (not at this bank)',
        # 'credit_history=delay in paying off in the past',
        # 'credit_history=existing credits paid back duly till now',
        # 'credit_history=no credits taken/ all credits paid back duly',
        # 'purpose=(vacation - does not exist?)',
        # 'purpose=business',
        # 'purpose=car (new)',
        # 'purpose=car (used)',
        # 'purpose=domestic appliances',
        # 'purpose=education',
        # 'purpose=furniture/equipment',
        # 'purpose=radio/television',
        # 'purpose=repairs',
        # 'purpose=retraining',
        'savings=.. >= 1000 DM ',
        'savings=... < 100 DM',
        'savings=100 <= ... < 500 DM',
        'savings=500 <= ... < 1000 DM ',
        'savings=unknown/ no savings account',
        'present_emp_since=.. >= 7 years',
        'present_emp_since=... < 1 year ',
        'present_emp_since=1 <= ... < 4 years',
        'present_emp_since=4 <= ... < 7 years',
        'present_emp_since=unemployed',
        # 'personal_status_sex=female : divorced/separated/married',
        # 'personal_status_sex=male : divorced/separated',
        # 'personal_status_sex=male : married/widowed',
        # 'personal_status_sex=male : single',
        'other_debtors=co-applicant',
        'other_debtors=guarantor',
        'other_debtors=none',
        'property=if not A121 : building society savings agreement/ life insurance',
        'property=if not A121/A122 : car or other, not in attribute 6',
        'property=real estate',
        'property=unknown / no property',
        'other_installment_plans=bank',
        'other_installment_plans=none',
        'other_installment_plans=stores',
        # 'housing=for free',
        # 'housing=own',
        # 'housing=rent',
        'job=management/ self-employed/ highly qualified employee/ officer',
        'job=skilled employee / official',
        'job=unemployed/ unskilled - non-resident',
        'job=unskilled - resident',
        'telephone=none',
        'telephone=yes, registered under the customers name ',
        # 'foreign_worker=no',
        # 'foreign_worker=yes',
    ],
    'home': [
        'beds',
        'bath',
        'price',
        # 'year_built',
        # 'sqft',
        'price_per_sqft',
        # 'elevation',
    ],
    'titanic': [
        'Pclass',
        # 'Age',
        # 'SibSp',
        # 'Parch',
        'Family',
        'Fare',
        # 'Sex=female',
        # 'Sex=male',
        'Embarked=C',
        'Embarked=Q',
        'Embarked=S',
    ],
    'iris': [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width'
    ],
    'moons': [
            'x0',
            'x1',
        ],
        'circles': [
                'x0',
                'x1',
            ],
        'linear': [
                'x0',
                'x1',
            ]

}

dataset_variable_features_names_noencode = {
    'adult': [
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'workclass',
        'occupation',
    ],
    'bank': [
        'revolving',
        'nbr_30_59_days_past_due_not_worse',
        'debt_ratio',
        'monthly_income',
        'nbr_open_credits_and_loans',
        'nbr_90_days_late',
        'nbr_real_estate_loans_or_lines',
        'nbr_60_89_days_past_due_not_worse',
        'dependents',
    ],
    'churn': [
        'account length',
        'number vmail messages',
        'total day minutes',
        'total day calls',
        'total day charge',
        'total eve minutes',
        'total eve calls',
        'total eve charge',
        'total night minutes',
        'total night calls',
        'total night charge',
        'total intl minutes',
        'total intl calls',
        'total intl charge',
        'customer service calls',
        'international plan=no',
        'international plan=yes',
        'voice mail plan=no',
        'voice mail plan=yes',
    ],
    'compas': [
        'priors_count',
        'days_b_screening_arrest',
        'is_recid',
        'is_violent_recid',
        'two_year_recid',
        'length_of_stay',
        'c_charge_degree=F',
        'c_charge_degree=M',
    ],
    'diabetes': [
        # 'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        # 'DiabetesPedigreeFunction',
        # 'Age',
    ],
    'fico': [
        # 'ExternalRiskEstimate',
        'MSinceOldestTradeOpen',
        'MSinceMostRecentTradeOpen',
        'AverageMInFile',
        'NumSatisfactoryTrades',
        'NumTrades60Ever2DerogPubRec',
        'NumTrades90Ever2DerogPubRec',
        'PercentTradesNeverDelq',
        'MSinceMostRecentDelq',
        'MaxDelq2PublicRecLast12M',
        'MaxDelqEver',
        'NumTotalTrades',
        'NumTradesOpeninLast12M',
        'PercentInstallTrades',
        'MSinceMostRecentInqexcl7days',
        'NumInqLast6M',
        'NumInqLast6Mexcl7days',
        'NetFractionRevolvingBurden',
        'NetFractionInstallBurden',
        'NumRevolvingTradesWBalance',
        'NumInstallTradesWBalance',
        'NumBank2NatlTradesWHighUtilization',
        'PercentTradesWBalance',
    ],
    'german': [
        'duration_in_month',
        'credit_amount',
        'installment_as_income_perc',
        'present_res_since',
        # 'age',
        'credits_this_bank',
        # 'people_under_maintenance',
        'account_check_status',
        # 'credit_history',
        # 'purpose',
        'savings',
        'present_emp_since',
        # 'personal_status_sex',
        'other_debtors',
        'property',
        'other_installment_plans',
        # 'housing',
        'job',
        'telephone',
        # 'foreign_worker',
    ],
    'home': [
        'beds',
        'bath',
        'price',
        # 'year_built',
        # 'sqft',
        'price_per_sqft',
        # 'elevation',
    ],
    'titanic': [
        'Pclass',
        # 'Age',
        # 'SibSp',
        # 'Parch',
        'Family',
        'Fare',
        # 'Sex',
        'Embarked',
    ],
    'iris': [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width'
    ],
    'moons': [
        'x0',
        'x1',
    ],
    'circles': [
            'x0',
            'x1',
        ],
    'linear': [
            'x0',
            'x1',
        ]
}


# def highlight_diff(data, color='red', x=None):
#     attr = 'color: %s' % color
#     is_diff = np.abs(df_x.values - data.values) != 0
#     return pd.DataFrame(np.where(is_diff, attr, ''),
#                         index=data.index, columns=data.columns)

def highlight_diff_first(data, color='red'):
    attr = 'color: %s' % color
    is_diff = data.values[:,0].reshape(-1,1) != data.values
    return pd.DataFrame(np.where(is_diff, attr, ''),
                        index=data.index, columns=data.columns)


def show_diff(x, cf_list, features_names, cont_features_names, cate_features_names,
              continuous_features_idx, categorical_features_lists_idx,
              color='red', scaler=None, y=None, y_proba=None, variable_features_names=None):
    df_x = pd.DataFrame(x.reshape(1, -1), columns=features_names)
    df_cf = pd.DataFrame(cf_list, columns=features_names)
    df_a = pd.concat([df_x, df_cf]).reset_index(drop=True)

    cols2remove = list()
    columns = np.array(df_a.columns)
    for cfn, indexes in zip(cate_features_names, categorical_features_lists_idx):
        values = df_a.iloc[:, indexes].idxmax(1).tolist()
        values = np.array([v.replace('%s=' % cfn, '') for v in values])
        df_a[cfn] = values
        cols2remove.extend(columns[indexes])
    df_a.drop(cols2remove, axis=1, inplace=True)
    if scaler:
        data = df_a.values
        data[:, continuous_features_idx] = scaler.inverse_transform(data[:, continuous_features_idx].astype(np.float64))
        df_a = pd.DataFrame(data=data, columns=df_a.columns)

    feat_names = cont_features_names + cate_features_names
    if variable_features_names:
        variable_features_names_real = set([c.split('=')[0] for c in variable_features_names])
        feat_names = [c for c in feat_names if c in variable_features_names_real]

    if y is not None:
        df_a['class'] = y
        feat_names += ['class']

    if y_proba is not None:
        df_a['prob'] = y_proba
        feat_names += ['prob']

    df_a_T = df_a[feat_names].T
    df_a_T.columns = ['x'] + ['cf_%i' % i for i in range(len(cf_list))]
    return df_a_T.style.apply(highlight_diff_first, color=color, axis=None)

# da rivedere
# def show_diff1h(x, cf_list, variable_features, variable_features_names, continuous_features_all,
#                 color='red', scaler=None):
#     df_x = pd.DataFrame(x.reshape(1,-1)[:, variable_features], columns=variable_features_names)
#     df_cf = pd.DataFrame(cf_list[:, variable_features], columns=variable_features_names)
#     df_a = pd.concat([df_x, df_cf]).reset_index(drop=True)
#     if scaler:
#         data = df_a.values
#         data[:,continuous_features_all] = scaler.inverse_transform(data[:,continuous_features_all].astype(np.float64))
#         df_a = pd.DataFrame(data=data, columns=df_a.columns)
#     df_a_T = df_a.T
#     df_a_T.columns = ['x'] + ['cf_%i' % i for i in range(len(cf_list))]
#     return df_a_T.style.apply(highlight_diff_first, color=color, axis=None)


def get_dict_record(x, features_names, continuous_features_all, categorical_features_lists_all, scaler):
    x_dict = {}

    values = scaler.inverse_transform(x[continuous_features_all].reshape(1,-1).astype(np.float64))[0]
    for a, v in zip(continuous_features_all, values):
        x_dict[features_names[a]] = v

    for indexes in categorical_features_lists_all:
        val = np.argmax(x[indexes])
        a = features_names[indexes[val]].split('=')[0]
        v = features_names[indexes[val]].split('=')[1]
        x_dict[a] = v

    return x_dict
