import platform

if 'Linux' in platform.platform():
    path = '/home/riccotti/Documents/Counterfactuals/'
    path = '/home/riccotti/Documents/CounterfactualExplanations/'
else:
    path = '/Users/riccardo/Documents/Research/CounterfactualExplanations/'

path_dataset = path + 'dataset/'
path_models = path + 'models/'
# path_results = path + 'results_gpu/'
# path_cf = path + 'cf_gpu/'
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

# Non ri-testare
# 'sace-rand-1000',
# 'sace-rand-100000',
# 'sace-feat-1-20', 'sace-feat-2-20',
# 'sace-neig-100', 'sace-clus-10',
# 'sace-tree-4f', 'sace-tree-8f', 'sace-tree-16f',

# Knonw train true /false non cambia nulla quindi mettiamo True
# search diversity va meglio senza quindi mettiamo False
# Distanza ok euclidean jaccard
