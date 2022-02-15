import platform

path_dataset = path + 'dataset/'
path_models = path + 'models/'
path_results = path + 'results/'
path_cf = path + 'cf/'
path_ae = path + 'aemodels/'
path_fig = path + 'fig/'

random_state = 0
test_size = 0.3

dataset_list = ['adult', 'bank', 'churn', 'compas', 'diabetes', 'fico', 'german', 'home', 'titanic',
                'mnist', 'fashion_mnist', 'cifar10', 'imagenet1000',
                'arrowhead', 'ecg200', 'ecg5000',  # 'diatom',
                'ecg5days', 'facefour', 'herring', 'gunpoint', 'italypower',
                'electricdevices', 'phalanges',
                'iris', 'moons', 'circles', 'linear'
                ]

blackbox_list = ['RF', 'NN', 'SVM', 'DNN', 'DNN2', 'LGBM',
                 'VGG16', 'VGG19', 'InceptionV3', 'ResNet50', 'InceptionResNetV2',
                 'CNN', 'ResNet', 'BiLSTM', 'LSTMFCN',
                 ]

cfe_list = ['sace-rand', 'sace-tree', 'sace-neig', 'sace-clus', 'sace-feat',
            'dice', 'mace', 'face', 'piece', 'esdce', 'chvae', 'cem', 'icegp', 'macem'
            ]
