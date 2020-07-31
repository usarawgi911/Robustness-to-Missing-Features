import numpy as np
np.random.seed(0)
import os
from dataset import load_dataset
# from evaluator import evaluate_n
import utils
import trainer
import models

config = utils.EasyDict({
    'dataset_dir': './datasets/',
    
    'dataset': 'wine',
    'model_dir': 'models_nips_20folds/',
    
    'n_folds': 20,
    
    # 'build_model': 'prorated',
    'build_model': 'point',

    'last_layer': 'concatenate',
    # 'last_layer': 'add',
    
    'mod_split' :'none',
    # 'mod_split' :'human',
    # 'mod_split' :'computation_split',
    # 'mod_split' :'random',

    'impute': 'mean',
    'drop': '0.3',
    'experiment': 2,
    'task': 'regression',
    # 'task': 'classification',

    'lr' : 0.1,
    # 'lr' : 0.1,

    'epochs' : 1000,
    
    'loss' : 'mse',
    
    # 'optimizer' : 'adam',

    # 'batch_size' : 32,
    'batch_size' : 100,

    'verbose': 0,
})


def main(config):

    config.expt_name = "Exp" + str(config.experiment) + "_" + config.mod_split + "_" + config.build_model + "_" + config.last_layer + "_lr" + str(config.lr) + "_bs" + str(config.batch_size) + "_epochs" + str(config.epochs) + "_folds" + str(config.n_folds)

    # Create save directories
    utils.create_directories(config)

    data = load_dataset(config)

    # data['0'] = np.asarray([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], dtype='float')
    # data['1'] = np.asarray([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype='float')
    # data['2'] = np.asarray([[1, 2], [1, 2], [1, 2]], dtype='float')
    # data['y'] = np.asarray([100, 200, 300], dtype='float')

    print(data.keys())

    if config.experiment==1 or config.experiment==3:
        n_feature_sets = len(data.keys()) - 1
    elif config.experiment==2:
        n_feature_sets = int(len(data.keys())/2) - 1
    
    X = [np.array(data['{}'.format(i)]) for i in range(n_feature_sets)]
    y = np.array(data['y'])

    if config.task=='classification':
        config.n_classes = len(set(y))

    if config.experiment==2:
        X_test = [np.array(data['{}_test'.format(i)]) for i in range(n_feature_sets)]
        y_test = np.array(data['y_test'])

    config.n_feature_sets = n_feature_sets
    config.feature_split_lengths = [i.shape[1] for i in X]

    print('Dataset used ', config.dataset)
    print('Number of feature sets ', n_feature_sets)
    [print('Shape of feature set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(X)]

    if config.experiment==2:
        [print('Shape of test feature set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(X_test)]

    if config.dataset in ['boston', 'cement', 'power_plant', 'wine', 'yacht', 'kin8nm', 'energy_efficiency', 'naval', 'life', 'pima']:
        config.units = 50
    elif config.dataset in ['msd', 'protein', 'toy']:
        config.units = 100
    else:
        config.units = 50

    if config.experiment==1:
        trainer.train_expt1(X, y, config)
    if config.experiment==2:
        trainer.train_expt2(X, y, X_test, y_test, config)
    if config.experiment==3:
        trainer.train_expt3(X, y, config)
    print(config.expt_name)
if __name__ == '__main__':
    main(config)