import numpy as np
import os
from dataset import load_dataset
# from evaluator import evaluate_n
import utils
import trainer

config = utils.EasyDict({
    'dataset_dir': './datasets/',
    
    'dataset': 'boston',
    'model_dir': 'models_test_scaled/',
    
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

    # 'lr' : 0.05,
    'lr' : 0.1,

    'epochs' : 3000,
    
    'loss' : 'mse',
    
    # 'optimizer' : 'adam',

    # 'batch_size' : 32,
    'batch_size' : 100,

    'verbose': 0,
})


def main(config):

    config.expt_name = config.mod_split + "_" + config.build_model + "_" + config.last_layer + "_lr" + str(config.lr) + "_bs" + str(config.batch_size) + "_epochs" + str(config.epochs)

    # Create save directories
    utils.create_directories(config)

    data = load_dataset(config)

    # data['0'] = np.asarray([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], dtype='float')
    # data['1'] = np.asarray([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype='float')
    # data['2'] = np.asarray([[1, 2], [1, 2], [1, 2]], dtype='float')
    # data['y'] = np.asarray([100, 200, 300], dtype='float')

    print(data.keys())

    n_feature_sets = len(data.keys()) - 1
    X = [np.array(data['{}'.format(i)]) for i in range(n_feature_sets)]
    y = np.array(data['y'])

    config.n_feature_sets = n_feature_sets
    config.feature_split_lengths = [i.shape[1] for i in X]

    print('Dataset used ', config.dataset)
    print('Number of feature sets ', n_feature_sets)
    [print('Shape of feature set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(X)]

    if config.dataset in ['boston', 'cement', 'power_plant', 'wine', 'yacht', 'kin8nm', 'energy_efficiency', 'naval']:
        config.units = 50
    elif config.dataset in ['msd', 'protein', 'toy']:
        config.units = 100

    trainer.train_expt(X, y, config)
    print(config.expt_name)
if __name__ == '__main__':
    main(config)