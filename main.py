import numpy as np
np.random.seed(0)
import os
from dataset import load_dataset
import utils
import trainer
import models

from opts import Opts

def main(config):

    if config.task=='train':
        config.train=1
    else:
        config.train=0

    if config.dataset=='life':
        config.task='regression'
    else:
        config.task='classification'
        config.experiment='doublecv'

    config.expt_name = "Exp" + str(config.experiment) + "_" + config.mod_split + "_" + config.build_model + "_" + config.last_layer
    
    # Create save directories
    utils.create_directories(config)
    data = load_dataset(config)

    if config.experiment=='mar_doublecv' or config.experiment=='doublecv':
        n_feature_sets = len(data.keys()) - 1
    elif config.dataset=='life':
        n_feature_sets = int(len(data.keys())/2) - 1
    
    X = [np.array(data['{}'.format(i)]) for i in range(n_feature_sets)]
    y = np.array(data['y'])

    X_test = None 
    y_test = None

    if config.task=='classification':
        config.n_classes = len(set(y))

    if config.dataset=='life':
        X_test = [np.array(data['{}_test'.format(i)]) for i in range(n_feature_sets)]
        y_test = np.array(data['y_test'])

    config.n_feature_sets = n_feature_sets
    config.feature_split_lengths = [i.shape[1] for i in X]

    if config.verbose>0:
        print('Dataset used ', config.dataset)
        print('Number of feature sets ', n_feature_sets)
        [print('Shape of feature set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(X)]
    
    trainer.train(X, y, config, X_test, y_test)

    print(config.expt_name)
    print(config.dataset)
if __name__ == '__main__':
    opts = Opts()
    config = opts.parse()
    main(config)