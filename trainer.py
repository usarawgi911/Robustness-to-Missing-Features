import os
import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

import models
import dataset
import utils

def train_expt(X, y, config):
	model, loss = models.build_model(config)
	indices = [i for i in range(len(y))]
	train_index, test_index, y_train, y_test = train_test_split(indices, y, test_size=0.2, random_state=43)
	X_train = [X[i][train_index] for i in range(config.n_feature_sets)]
	X_test = [X[i][test_index] for i in range(config.n_feature_sets)]

	X_train, X_test = utils.drop_features(X_train, X_test, config)

	checkpoint_filepath = os.path.join(config.model_dir, 'test_boston_1.h5')
	checkpointer = tf.keras.callbacks.ModelCheckpoint(
			checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True,
			save_weights_only=True, mode='auto', save_freq='epoch')

	model.compile(optimizer=tf.optimizers.Adam(learning_rate=config.lr),
					  loss=loss)
	history = model.fit(X_train, y_train,
						batch_size=config.batch_size,
						epochs=config.epochs,
						verbose=config.verbose,
						callbacks=[checkpointer],
						validation_data=(X_test, y_test))
	print(np.sqrt(min(history.history['val_loss'])))
	# print(history)
	return