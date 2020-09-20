import os
import numpy as np
np.random.seed(0)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_boston
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
import utils

def random_split(config, features):
    data = np.transpose(features)
    clusters = feature_split(config, features, return_split_sizes=True)
    n_features = len(data)
    
    rand_range = [x for x in range(n_features)]
    np.random.shuffle(rand_range)
    
    X = []
    ind=0
    
    for c in set(clusters):
        cluster_size = list(clusters).count(c)
        indices = rand_range[ind:ind+cluster_size]
        # print(indices)
        X.append(np.transpose(data[indices]))
        ind+=cluster_size
    return X

def feature_split(config, features, return_split_sizes=False):
    from scipy.cluster.hierarchy import linkage
    data = np.transpose(features)

    ######### Hierarchical Clustering based on correlation
    Y = pdist(data, 'correlation')

    linkage = linkage(Y, 'complete')
    
    # dendrogram(linkage, color_threshold=0)
    # plt.show()
    if config.dataset == 'energy_efficiency':
        linkage = np.load('energy_efficiency_linkage.npy')
    if config.dataset=='msd':
        clusters = fcluster(linkage, 0.75 * Y.max(), 'distance')
    else:
        clusters = fcluster(linkage, config.hc_threshold* Y.max(), 'distance')

    if(return_split_sizes):
        return clusters

    X = []
    for cluster in set(clusters):
        indices = [j for j in range(len(clusters)) if clusters[j]==cluster]
        # print(indices)
        X.append(np.transpose(data[indices]))
    return X

def feature_as_a_cluster(config, features):
    X = []
    for idx in range(features.shape[-1]):
        X.append(features[:,idx].reshape(-1,1))
    return X

def load_dataset(config):
    np.random.seed(0)
    
    if config.dataset not in ['life'] and (config.experiment==4 or config.experiment=='doublecv'):
        data = _nips2019datasets_cv(config)

    elif config.dataset=='boston':
        data = _boston(config)        

    elif config.dataset=='cement':
        data = _cement(config)

    elif config.dataset=='energy_efficiency':
        data = _energy_efficiency(config)

    elif config.dataset=='kin8nm':
        data = _kin8nm(config)

    elif config.dataset=='power_plant':
        data = _power_plant(config)

    elif config.dataset=='protein':
        data = _protein(config)

    elif config.dataset=='wine':
        data = _wine(config)

    elif config.dataset=='yacht':
        data = _yacht(config)
    
    elif config.dataset=='naval':
        data = _naval(config)

    elif config.dataset=='msd':
        data = _msd(config)

    elif config.dataset=='life':
        data = _life(config)

    elif config.dataset=='pima':
        data = _pima(config)

    elif config.dataset=='horse':
        data = _horse(config)

    elif config.dataset=='bands':
        data = _bands(config)

    elif config.dataset=='hepatitis':
        data = _hepatitis(config)

    elif config.dataset=='mammographics':
        data = _mammographics(config)

    elif config.dataset=='kidney_disease':
        data = _kidney_disease(config)

    elif config.dataset=='winconsin':
        data = _winconsin(config)

    elif config.dataset=='esr':
        data = _esr(config)


    return data


def _boston(config):
    data_df = load_boston()
    df = pd.DataFrame(data=data_df['data'], columns=data_df['feature_names'])
    y = data_df['target']

    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y}

    elif config.mod_split=='human':
        features1 = ['ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX']
        features2 = ['CRIM', 'PTRATIO', 'B', 'LSTAT']
        X1 = df[features1].values
        X2 = df[features2].values
        data = {'0':X1, '1':X2, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _cement(config):
    data_df = pd.read_csv(os.path.join(config.dataset_dir, 'cement.csv'))
    
    target_col = 'Concrete compressive strength(MPa, megapascals) '
    y = data_df[target_col]

    df = data_df.drop(columns=[target_col])
    
    if config.mod_split=='none' or config.mod_split=='human': # since only 1 split
        X = df.values
        data = {'0':X, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _energy_efficiency(config):

    data_df = pd.read_csv(os.path.join(config.dataset_dir, 'energy_efficiency.csv'))
    
    target_col1 = 'Heating Load'
    target_col2 = 'Cooling Load'
    y = data_df[target_col1]

    df = data_df.drop(columns=[target_col1, target_col2])

    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y}

    if config.mod_split=='human':
        # X5
        # X1, X2, X3, X4
        # X7, X8
        # X6

        features1 = ['Overall Height']
        features2 = ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area']
        features3 = ['Glazing Area', 'Glazing Area Distribution']
        features4 = ['Orientation']

        X1 = df[features1].values
        X2 = df[features2].values
        X3 = df[features3].values
        X4 = df[features4].values
        data = {'0':X1, '1':X2, '2':X3, '3':X4, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _kin8nm(config):
    data_df = pd.read_csv(os.path.join(config.dataset_dir, 'kin8nm.csv'))
    
    y = data_df['y']

    df = data_df.drop(columns=['y'])
    if config.mod_split=='none' or config.mod_split=='human': # since only 1 split
        X = df.values
        data = {'0':X, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _power_plant(config):
    data_df = pd.read_csv(os.path.join(config.dataset_dir, 'power_plant.csv'))
    
    y = data_df['PE']

    df = data_df.drop(columns=['PE'])
    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y}

    elif config.mod_split=='human':
        # T, AP, RH
        # V

        features1 = ['AT', 'AP', 'RH']
        features2 = ['V']
        X1 = df[features1].values
        X2 = df[features2].values
        data = {'0':X1, '1':X2, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _protein(config):
    data_df = pd.read_csv(os.path.join(config.dataset_dir, 'protein.csv'))
    
    y = data_df['RMSD']

    df = data_df.drop(columns=['RMSD'])

    if config.mod_split=='none' or config.mod_split=='human':
        X = df.values
        data = {'0':X, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _wine(config):
    data_df = pd.read_csv(os.path.join(config.dataset_dir, 'wine.csv'))
    
    y = data_df['quality']

    df = data_df.drop(columns=['quality'])

    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y}

    if config.mod_split=='human':
        features1 = ['alcohol', 'pH', 'fixed acidity', 'density', 'residual sugar']
        features2 = ['volatile acidity', 'citric acid']
        features3 = ['chlorides','free sulfur dioxide', 'total sulfur dioxide', 'sulphates']

        X1 = df[features1].values
        X2 = df[features2].values
        X3 = df[features3].values
        data = {'0':X1, '1':X2, '2':X3, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _yacht(config):
    cols = ['Longitudinal position of the center of buoyancy', 'Prismatic coefficient', 'Length-displacement ratio', 
            'Beam-draught ratio', 'Length-beam ratio', 'Froude number', 'Residuary resistance per unit weight of displacement']

    data_df = pd.read_csv(os.path.join(config.dataset_dir, 'yacht.data'), sep="\\s+", names=cols)
    target_col = 'Residuary resistance per unit weight of displacement'
    y = data_df[target_col]
    df = data_df.drop(columns=[target_col])
    if config.mod_split=='none' or config.mod_split=='human':
        X = df.values
        data = {'0':X, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _naval(config):
    cols = ['lp', 'v', 'gtt', 'gtn', 'ggn', 'ts', 'tp',
            't48', 't1', 't2', 'p48', 'p1', 'p2', 'pexh',
            'tic', 'mf', 'y1', 'y2']

    data_df = pd.read_csv(os.path.join(config.dataset_dir, 'naval.csv'), sep='\\s+', names=cols)
    target_col = ['y1', 'y2']
    
    y = data_df[target_col[0]]

    df = data_df.drop(columns=['y1', 'y2', 't1', 'p1'])

    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y}

    if config.mod_split=='human':
        features1 = ['t48', 't2']
        features2 = ['p48', 'p2', 'pexh']
        features3 = ['gtt', 'gtn', 'ggn', 'ts', 'tp']
        features4 = ['lp', 'v', 'tic', 'mf']

        X1 = df[features1].values
        X2 = df[features2].values
        X3 = df[features3].values
        X4 = df[features4].values
        data = {'0':X1, '1':X2, '2':X3, '3':X4, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _msd(config):
    
    data_df = pd.read_csv(os.path.join(config.dataset_dir, 'year_prediction.csv'))
    target_col = 'label'
    
    y = data_df[target_col]

    df = data_df.drop(columns=['label'])
    cols = df.columns.tolist()
    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y}

    if config.mod_split=='human':
        features1 = cols[:12]
        features2 = cols[12:]

        X1 = df[features1].values
        X2 = df[features2].values
        data = {'0':X1, '1':X2, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _life(config):
    data_df = pd.read_csv(os.path.join(config.dataset_dir, 'life_expectancy.csv'))
    data_df[['Country']] = data_df[['Country']].apply(LabelEncoder().fit_transform)
    data_df[['Status']] = data_df[['Status']].apply(LabelEncoder().fit_transform)
    data_df_dropped = data_df.dropna()
    
    target_col = 'Life expectancy '
    ###

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()

    data_df = data_df[~data_df['Life expectancy '].isnull()]
    null_df = data_df[data_df.isnull().any(axis=1)]
    filled_df = null_df.fillna(np.nan)

    y_test = np.asarray(filled_df[target_col])
    x_test = filled_df.drop(columns=[target_col]).values
    ###

    
    y = np.asarray(data_df_dropped[target_col])

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()
    
    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y, '0_test':x_test, 'y_test':y_test}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

        clusters = feature_split(config, df.values, return_split_sizes=True)
        X_test_split = []
        for cluster in set(clusters):
            indices = [j for j in range(len(clusters)) if clusters[j]==cluster]
            print(indices)
            X_test_split.append(x_test[:, indices])
        
        for i, x in enumerate(X_test_split):
            data["{}_test".format(i)] = x

        data['y_test'] = y_test
    return data

def _pima(config):
    tmp_df1 = pd.DataFrame(utils.read_data(config.dataset_dir+'/pima_data.txt'))
    tmp_df2 = pd.DataFrame(utils.read_data(config.dataset_dir+'/pima_labels.txt'))
    data_df = pd.concat([tmp_df1, tmp_df2], axis=1)
    data_df.columns = [str(x) for x in range(len(data_df.columns))]
    target_col = '8'
    data_df[[target_col]] = data_df[[target_col]].replace(-1, 0)

    data_df_dropped = data_df.dropna()
    
    ###

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()

    data_df = data_df[~data_df[target_col].isnull()]
    null_df = data_df[data_df.isnull().any(axis=1)]
    filled_df = null_df.fillna(np.nan)

    y_test = np.asarray(filled_df[target_col])
    x_test = filled_df.drop(columns=[target_col]).values
    ###

    
    y = np.asarray(data_df_dropped[target_col])

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()
    print("x_test {}, y_test {}".format(x_test.shape, y_test.shape))
    print("df {}, y {}".format(df.shape, y.shape))
    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y, '0_test':x_test, 'y_test':y_test}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

        clusters = feature_split(config, df.values, return_split_sizes=True)
        X_test_split = []
        for cluster in set(clusters):
            indices = [j for j in range(len(clusters)) if clusters[j]==cluster]
            print(indices)
            X_test_split.append(x_test[:, indices])
        
        for i, x in enumerate(X_test_split):
            data["{}_test".format(i)] = x

        data['y_test'] = y_test
    return data

def _bands(config):
    tmp_df1 = pd.DataFrame(utils.read_data(config.dataset_dir+'/bands_data.txt'))
    tmp_df2 = pd.DataFrame(utils.read_data(config.dataset_dir+'/bands_labels.txt'))
    data_df = pd.concat([tmp_df1, tmp_df2], axis=1)
    data_df.columns = [str(x) for x in range(len(data_df.columns))]
    target_col = str(len(data_df.columns)-1)
    data_df[[target_col]] = data_df[[target_col]].replace(-1, 0)

    data_df_dropped = data_df.dropna()
    
    ###

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()

    data_df = data_df[~data_df[target_col].isnull()]
    null_df = data_df[data_df.isnull().any(axis=1)]
    filled_df = null_df.fillna(np.nan)

    y_test = np.asarray(filled_df[target_col])
    x_test = filled_df.drop(columns=[target_col]).values
    ###

    
    y = np.asarray(data_df_dropped[target_col])

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()
    print("x_test {}, y_test {}".format(x_test.shape, y_test.shape))
    print("df {}, y {}".format(df.shape, y.shape))
    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y, '0_test':x_test, 'y_test':y_test}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

        clusters = feature_split(config, df.values, return_split_sizes=True)
        X_test_split = []
        for cluster in set(clusters):
            indices = [j for j in range(len(clusters)) if clusters[j]==cluster]
            print(indices)
            X_test_split.append(x_test[:, indices])
        
        for i, x in enumerate(X_test_split):
            data["{}_test".format(i)] = x

        data['y_test'] = y_test
    return data

def _horse(config):
    tmp_df1 = pd.DataFrame(utils.read_data(config.dataset_dir+'/horse_data.txt'))
    tmp_df2 = pd.DataFrame(utils.read_data(config.dataset_dir+'/horse_labels.txt'))
    data_df = pd.concat([tmp_df1, tmp_df2], axis=1)
    data_df.columns = [str(x) for x in range(len(data_df.columns))]
    target_col = str(len(data_df.columns)-1)
    data_df[[target_col]] = data_df[[target_col]].replace(-1, 0)

    data_df_dropped = data_df.dropna()
    
    ###

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()

    data_df = data_df[~data_df[target_col].isnull()]
    null_df = data_df[data_df.isnull().any(axis=1)]
    filled_df = null_df.fillna(np.nan)

    y_test = np.asarray(filled_df[target_col])
    x_test = filled_df.drop(columns=[target_col]).values
    ###

    
    y = np.asarray(data_df_dropped[target_col])

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()
    print("x_test {}, y_test {}".format(x_test.shape, y_test.shape))
    print("df {}, y {}".format(df.shape, y.shape))
    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y, '0_test':x_test, 'y_test':y_test}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

        clusters = feature_split(config, df.values, return_split_sizes=True)
        X_test_split = []
        for cluster in set(clusters):
            indices = [j for j in range(len(clusters)) if clusters[j]==cluster]
            print(indices)
            X_test_split.append(x_test[:, indices])
        
        for i, x in enumerate(X_test_split):
            data["{}_test".format(i)] = x

        data['y_test'] = y_test
    return data

def _hepatitis(config):
    tmp_df1 = pd.DataFrame(utils.read_data(config.dataset_dir+'/hepatitis_data.txt'))
    tmp_df2 = pd.DataFrame(utils.read_data(config.dataset_dir+'/hepatitis_labels.txt'))
    data_df = pd.concat([tmp_df1, tmp_df2], axis=1)
    data_df.columns = [str(x) for x in range(len(data_df.columns))]
    target_col = str(len(data_df.columns)-1)
    data_df[[target_col]] = data_df[[target_col]].replace(-1, 0)

    data_df_dropped = data_df.dropna()
    
    ###

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()

    data_df = data_df[~data_df[target_col].isnull()]
    null_df = data_df[data_df.isnull().any(axis=1)]
    filled_df = null_df.fillna(np.nan)

    y_test = np.asarray(filled_df[target_col])
    x_test = filled_df.drop(columns=[target_col]).values
    ###

    
    y = np.asarray(data_df_dropped[target_col])

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()
    print("x_test {}, y_test {}".format(x_test.shape, y_test.shape))
    print("df {}, y {}".format(df.shape, y.shape))
    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y, '0_test':x_test, 'y_test':y_test}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

        clusters = feature_split(config, df.values, return_split_sizes=True)
        X_test_split = []
        for cluster in set(clusters):
            indices = [j for j in range(len(clusters)) if clusters[j]==cluster]
            print(indices)
            X_test_split.append(x_test[:, indices])
        
        for i, x in enumerate(X_test_split):
            data["{}_test".format(i)] = x

        data['y_test'] = y_test
    return data

def _hepatitis(config):
    tmp_df1 = pd.DataFrame(utils.read_data(config.dataset_dir+'/hepatitis_data.txt'))
    tmp_df2 = pd.DataFrame(utils.read_data(config.dataset_dir+'/hepatitis_labels.txt'))
    data_df = pd.concat([tmp_df1, tmp_df2], axis=1)
    data_df.columns = [str(x) for x in range(len(data_df.columns))]
    target_col = str(len(data_df.columns)-1)
    data_df[[target_col]] = data_df[[target_col]].replace(-1, 0)

    data_df_dropped = data_df.dropna()
    
    ###

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()

    data_df = data_df[~data_df[target_col].isnull()]
    null_df = data_df[data_df.isnull().any(axis=1)]
    filled_df = null_df.fillna(np.nan)

    y_test = np.asarray(filled_df[target_col])
    x_test = filled_df.drop(columns=[target_col]).values
    ###

    
    y = np.asarray(data_df_dropped[target_col])

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()
    print("x_test {}, y_test {}".format(x_test.shape, y_test.shape))
    print("df {}, y {}".format(df.shape, y.shape))
    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y, '0_test':x_test, 'y_test':y_test}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

        clusters = feature_split(config, df.values, return_split_sizes=True)
        X_test_split = []
        for cluster in set(clusters):
            indices = [j for j in range(len(clusters)) if clusters[j]==cluster]
            print(indices)
            X_test_split.append(x_test[:, indices])
        
        for i, x in enumerate(X_test_split):
            data["{}_test".format(i)] = x

        data['y_test'] = y_test
    return data

def _mammographics(config):
    tmp_df1 = pd.DataFrame(utils.read_data(config.dataset_dir+'/mammographics_data.txt'))
    tmp_df2 = pd.DataFrame(utils.read_data(config.dataset_dir+'/mammographics_labels.txt'))
    data_df = pd.concat([tmp_df1, tmp_df2], axis=1)
    data_df.columns = [str(x) for x in range(len(data_df.columns))]
    target_col = str(len(data_df.columns)-1)
    data_df[[target_col]] = data_df[[target_col]].replace(-1, 0)

    data_df_dropped = data_df.dropna()
    
    ###

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()

    data_df = data_df[~data_df[target_col].isnull()]
    null_df = data_df[data_df.isnull().any(axis=1)]
    filled_df = null_df.fillna(np.nan)

    y_test = np.asarray(filled_df[target_col])
    x_test = filled_df.drop(columns=[target_col]).values
    ###

    
    y = np.asarray(data_df_dropped[target_col])

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()
    print("x_test {}, y_test {}".format(x_test.shape, y_test.shape))
    print("df {}, y {}".format(df.shape, y.shape))
    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y, '0_test':x_test, 'y_test':y_test}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

        clusters = feature_split(config, df.values, return_split_sizes=True)
        X_test_split = []
        for cluster in set(clusters):
            indices = [j for j in range(len(clusters)) if clusters[j]==cluster]
            print(indices)
            X_test_split.append(x_test[:, indices])
        
        for i, x in enumerate(X_test_split):
            data["{}_test".format(i)] = x

        data['y_test'] = y_test
    return data

def _kidney_disease(config):
    tmp_df1 = pd.DataFrame(utils.read_data(config.dataset_dir+'/kidney_disease_data.txt'))
    tmp_df2 = pd.DataFrame(utils.read_data(config.dataset_dir+'/kidney_disease_labels.txt'))
    data_df = pd.concat([tmp_df1, tmp_df2], axis=1)
    data_df.columns = [str(x) for x in range(len(data_df.columns))]
    target_col = str(len(data_df.columns)-1)
    data_df[[target_col]] = data_df[[target_col]].replace(-1, 0)

    data_df_dropped = data_df.dropna()
    
    ###

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()

    data_df = data_df[~data_df[target_col].isnull()]
    null_df = data_df[data_df.isnull().any(axis=1)]
    filled_df = null_df.fillna(np.nan)

    y_test = np.asarray(filled_df[target_col])
    x_test = filled_df.drop(columns=[target_col]).values
    ###

    
    y = np.asarray(data_df_dropped[target_col])

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()
    print("x_test {}, y_test {}".format(x_test.shape, y_test.shape))
    print("df {}, y {}".format(df.shape, y.shape))
    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y, '0_test':x_test, 'y_test':y_test}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

        clusters = feature_split(config, df.values, return_split_sizes=True)
        X_test_split = []
        for cluster in set(clusters):
            indices = [j for j in range(len(clusters)) if clusters[j]==cluster]
            print(indices)
            X_test_split.append(x_test[:, indices])
        
        for i, x in enumerate(X_test_split):
            data["{}_test".format(i)] = x

        data['y_test'] = y_test
    return data

def _winconsin(config):
    tmp_df1 = pd.DataFrame(utils.read_data(config.dataset_dir+'/winconsin_data.txt'))
    tmp_df2 = pd.DataFrame(utils.read_data(config.dataset_dir+'/winconsin_labels.txt'))
    data_df = pd.concat([tmp_df1, tmp_df2], axis=1)
    data_df.columns = [str(x) for x in range(len(data_df.columns))]
    target_col = str(len(data_df.columns)-1)
    data_df[[target_col]] = data_df[[target_col]].replace(-1, 0)

    data_df_dropped = data_df.dropna()
    
    ###

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()

    data_df = data_df[~data_df[target_col].isnull()]
    null_df = data_df[data_df.isnull().any(axis=1)]
    filled_df = null_df.fillna(np.nan)

    y_test = np.asarray(filled_df[target_col])
    x_test = filled_df.drop(columns=[target_col]).values
    ###

    
    y = np.asarray(data_df_dropped[target_col])

    df = data_df_dropped.drop(columns=[target_col])
    cols = df.columns.tolist()
    print("x_test {}, y_test {}".format(x_test.shape, y_test.shape))
    print("df {}, y {}".format(df.shape, y.shape))
    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y, '0_test':x_test, 'y_test':y_test}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

        clusters = feature_split(config, df.values, return_split_sizes=True)
        X_test_split = []
        for cluster in set(clusters):
            indices = [j for j in range(len(clusters)) if clusters[j]==cluster]
            print(indices)
            X_test_split.append(x_test[:, indices])
        
        for i, x in enumerate(X_test_split):
            data["{}_test".format(i)] = x

        data['y_test'] = y_test
    return data


def _nips2019datasets_cv(config):
    tmp_df1 = pd.DataFrame(utils.read_data(config.dataset_dir+'/' + config.dataset +'_data.txt'))
    tmp_df2 = pd.DataFrame(utils.read_data(config.dataset_dir+'/' + config.dataset +'_labels.txt'))
    data_df = pd.concat([tmp_df1, tmp_df2], axis=1)
    data_df.columns = [str(x) for x in range(len(data_df.columns))]
    target_col = str(len(data_df.columns)-1)
    data_df[[target_col]] = data_df[[target_col]].replace(-1, 0)

    df = data_df.drop(columns=[target_col])
    df = df.fillna(np.nan)
    cols = df.columns.tolist()
    y = np.asarray(data_df[target_col])
    print("df {}, y {}".format(df.shape, y.shape))

    if config.mod_split=='none':
        X = df.values
        X, _ = utils.replace_missing(X, X, config, True)
        data = {'0':X, 'y':y}

    elif config.mod_split=='computation_split':
        X = df.values
        X_filled, _ = utils.replace_missing(df.values, df.values, config, True)
        clusters = feature_split(config, X_filled, return_split_sizes=True)
        data = {'y':y}
        
        X_split = []
        for cluster in set(clusters):
            indices = [j for j in range(len(clusters)) if clusters[j]==cluster]
            print(indices)
            X_split.append(X[:, indices])
        
        for i, x in enumerate(X_split):
            data[str(i)] = x        

    return data

def _esr(config):
    data_df = pd.read_csv(os.path.join(config.dataset_dir, 'esr.csv'))
    data_df = data_df.drop(columns=['Unnamed: 0'])
    
    target_col = 'y'

    data_df[[target_col]] = data_df[[target_col]] - 1
    y = data_df[target_col]

    df = data_df.drop(columns=[target_col])
    df = df.astype(float)
    cols = df.columns.tolist()
    
    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data