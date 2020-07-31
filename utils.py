import numpy as np
np.random.seed(0)
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler

# Config to choose the hyperparameters for everything
class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name] 

def create_directories(config):
    model_dir = Path(config.model_dir)
    model_dir.joinpath(config.dataset).mkdir(parents=True, exist_ok=True)  

    model_dir = Path(os.path.join(config.model_dir, config.dataset))
    model_dir.joinpath(config.expt_name).mkdir(parents=True, exist_ok=True)

def replace_missing(X_train, X_test, config):
	if config.impute=='mean':
		col_means = [0]*config.n_feature_sets
		for i in range(config.n_feature_sets):
			ind = np.where(np.isnan(X_train[i]))
			col_mean = np.nanmean(X_train[i], axis=0)
			X_train[i][ind] = np.take(col_mean, ind[1])
			col_means[i] = col_mean

		for i in range(config.n_feature_sets):
			ind = np.where(np.isnan(X_test[i]))
			# col_mean = np.nanmean(X_test[i], axis=0)
			col_mean = col_means[i]
			X_test[i][ind] = np.take(col_mean, ind[1])

	return X_train, X_test

def drop_features(X_train, X_test, config):
	np.random.seed(0)
	
	if config.experiment==1:
		return drop_features_random(X_train, X_test, config)
	elif config.experiment==2:
		return replace_missing(X_train, X_test, config)

def drop_features_random(X_train, X_test, config):
	miss_perc = float(config.drop)
	features = np.sum(config.feature_split_lengths)
	rows_train = len(X_train[0])
	rows_test = len(X_test[0])
	
	train_indices = np.random.choice(rows_train*features, np.floor(rows_train*features*miss_perc).astype(np.int), replace=False)
	train_pattern = np.asarray([0]*rows_train*features)
	train_pattern[train_indices] = 1
	miss_train = train_pattern.reshape((rows_train, features))

	test_indices = np.random.choice(rows_test*features, np.floor(rows_test*features*miss_perc).astype(np.int), replace=False)
	test_pattern = np.asarray([0]*rows_test*features)
	test_pattern[test_indices] = 1
	miss_test = test_pattern.reshape((rows_test, features))

	cur=0
	for i, feature_len in enumerate(config.feature_split_lengths):

		train_features_indices = np.where(miss_train[:, cur:cur+config.feature_split_lengths[i]-1].flatten()==1)[0]
		cur_shape = X_train[i].shape
		tmp_X_train = np.asarray(X_train[i].flatten())
		# print("tmp_X_train {}, train_features_indices {}".format(tmp_X_train, train_features_indices))
		tmp_X_train[train_features_indices] = np.nan
		X_train[i] = tmp_X_train.reshape(cur_shape)

		test_features_indices = np.where(miss_test[:, cur:cur+config.feature_split_lengths[i]-1].flatten()==1)[0]
		cur_shape = X_test[i].shape
		tmp_X_test = X_test[i].flatten()
		tmp_X_test[test_features_indices] = np.nan
		X_test[i] = tmp_X_test.reshape(cur_shape)
		
		# X_train[i], X_test[i] = standard_scale(X_train[i], X_test[i])
	
		# X_test[i][test_indices[cur:cur+config.feature_split_lengths-1]] = np.nan
		cur+=config.feature_split_lengths[i]
	# print("Dropped features")
	return replace_missing(X_train, X_test, config)

def standard_scale(x_train, x_test):
	scalar = StandardScaler()
	scalar.fit(x_train)
	x_train = scalar.transform(x_train)
	x_test = scalar.transform(x_test)
	return x_train, x_test

def read_data(path, sep=',', val_type='f8'):
	return np.genfromtxt(path, dtype=val_type, delimiter=sep)
