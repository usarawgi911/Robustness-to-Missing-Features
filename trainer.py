import os
import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score

import models
import dataset
import utils

def train(X, y, config, X_test=None, y_test=None):

	if config.dataset=='life':
		train_life(X, y, config, X_test, y_test)
	elif config.experiment=='mar_doublecv' or config.experiment=='doublecv':
		train_doublecv(X, y, config)

def train_doublecv(X, y, config):
	print("In train_mar_doublecv")
	train_scores, val_scores, test_scores, best_epochs = [], [], [], []
	reports = []

	n_feature_sets = len(X)
	skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=42)
	fold = 1
	for train_val_index, test_index in skf.split(y, y):
		y_train_val, y_test = y[train_val_index], y[test_index]
		x_train_val = [i[train_val_index] for i in X]
		x_test = [i[test_index] for i in X]

		train_index, val_index = next(StratifiedKFold(n_splits=5, random_state=42).split(y_train_val, y_train_val))

		if config.experiment=="mar_doublecv":
			x_train_val, x_test = utils.drop_features(x_train_val, x_test, config)

		y_train, y_val = y_train_val[train_index], y_train_val[val_index]
		x_train = [i[train_index] for i in x_train_val]
		x_val = [i[val_index] for i in x_train_val]

		y_train_ohe, y_val_ohe, y_test_ohe = utils.ohe(y_train, y_val, y_test)

		print('Fold {}'.format(fold))

		# for i in range(n_feature_sets):
		# 	x_train[i], x_val[i] = utils.standard_scale(x_train[i], x_val[i])

		if config.verbose>0:
			print("train {}".format(np.unique(y_train, return_counts=True)))
			print("val {}".format(np.unique(y_val, return_counts=True)))
			print("test {}".format(np.unique(y_test, return_counts=True)))

		model, loss = models.build_model(config)

		checkpoint_filepath = os.path.join(config.model_dir, config.dataset, config.expt_name, 'model_'+str(fold) )
		checkpointer = tf.keras.callbacks.ModelCheckpoint(
				checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True,
				save_weights_only=True, mode='auto', save_freq='epoch')

		# tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/{}@{}@{}'.format(str(config.dataset), str(config.expt_name), 
		# 											 str(fold)), histogram_freq=1, 
		# 											 write_graph=True, write_images=False)

		if config.train:
			model.compile(optimizer=tf.optimizers.Adam(learning_rate=config.lr),
						  loss=loss, metrics=['accuracy'])	
		else:
			model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-2),
						  loss=loss, metrics=['accuracy'])	

		# [print('Shape of feature train set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(x_train)]
		# [print('Shape of feature val set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(x_val)]
		# [print('Shape of feature test set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(x_test)]
		
		# print('Shape of y train set {}'.format(np.shape(y_train_ohe)))
		# print('Shape of y val set {}'.format(np.shape(y_val_ohe)))
		# print('Shape of y test set {}'.format(np.shape(y_test_ohe)))
		
		if config.train:
			history = model.fit(x_train, y_train_ohe,
								batch_size=config.batch_size,
								epochs=config.epochs,
								verbose=config.verbose,
								validation_data=(x_val, y_val_ohe),
								callbacks=[checkpointer])
			model.load_weights(checkpoint_filepath)

			best_epoch = np.argmin(history.history['val_loss'])+1
			train_score = history.history['accuracy'][best_epoch-1]
			val_score = history.history['val_accuracy'][best_epoch-1]

		else:
			model.load_weights(checkpoint_filepath).expect_partial()
		y_preds = np.argmax(model.predict(x_test), axis=1)
		test_score = accuracy_score(y_test, y_preds)
		# reports.append(classification_report(y_test, y_preds))
		# print(classification_report(y_test, y_preds))

		if config.train :
			print("Best epoch {}, Train score {}, Val score {}".format(best_epoch, train_score, val_score))
			train_scores.append(train_score)
			val_scores.append(val_score)
			best_epochs.append(best_epoch)
		test_scores.append(test_score)
		fold+=1

	print("Results :")
	if config.train :
		print("best_epochs {}".format(best_epochs))
		print("train_scores {:.3f}+/-{:.3f} : {}".format(np.mean(train_scores), np.std(train_scores), np.round(train_scores, 3)))
		print("val_scores {:.3f}+/-{:.3f} : {}".format(np.mean(val_scores), np.std(val_scores), np.round(val_scores, 3)))
	print("test_scores {:.3f}+/-{:.3f} : {}".format(np.mean(test_scores), np.std(test_scores), np.round(test_scores, 3)))

	if config.train :
		print("\n{:.3f}+/-{:.3f}	{:.3f}+/-{:.3f}	{:.3f}+/-{:.3f}	{}	{}	{}\n".format(np.mean(train_scores), np.std(train_scores), 
																					np.mean(val_scores), np.std(val_scores),
																					np.mean(test_scores), np.std(test_scores),
																					np.round(train_scores, 3),
																					np.round(val_scores, 3),
																					np.round(test_scores, 3),
																					))

def train_life(X, y, config, x_test, y_test):
	print("In train life")

	fold = 1

	train_scores, val_scores, test_scores, best_epochs = [], [], [], []
	reports = []

	n_feature_sets = len(X)
	
	y_train_val = y
	x_train_val = X

	train_index, val_index = next(KFold(n_splits=5, random_state=42).split(y_train_val, y_train_val))

	y_train, y_val = y_train_val[train_index], y_train_val[val_index]
	x_train = [i[train_index] for i in x_train_val]
	x_val = [i[val_index] for i in x_train_val]

	_, X_test = utils.replace_missing(X, x_test, config, False)

	for i in range(n_feature_sets):
		x_train[i], x_val[i] = utils.standard_scale(x_train[i], x_val[i])
		_, x_test[i] = utils.standard_scale(x_train_val[i], x_test[i])

	# for i in range(n_feature_sets):
	# 	print(np.mean(x_train[i]), np.mean(x_val[i]), np.mean(x_test[i]))

	model, loss = models.build_model(config)

	checkpoint_filepath = os.path.join(config.model_dir, config.dataset, config.expt_name, 'model')
	
	checkpointer = tf.keras.callbacks.ModelCheckpoint(
			checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True,
			save_weights_only=True, mode='auto', save_freq='epoch')

	# tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/{}@{}@{}'.format(str(config.dataset), str(config.expt_name), 
	# 											 str(fold)), histogram_freq=1, 
	# 											 write_graph=True, write_images=False)

	if config.train:
		model.compile(optimizer=tf.optimizers.Adam(learning_rate=config.lr),
					  loss=loss)
	else:
		model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-2),
					  loss=loss)
	# [print('Shape of feature train set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(x_train)]
	# [print('Shape of feature val set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(x_val)]
	# [print('Shape of feature test set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(x_test)]
	
	# print('Shape of y train set {}'.format(np.shape(y_train_ohe)))
	# print('Shape of y val set {}'.format(np.shape(y_val_ohe)))
	# print('Shape of y test set {}'.format(np.shape(y_test_ohe)))
	
	if config.train:
		history = model.fit(x_train, y_train,
							batch_size=config.batch_size,
							epochs=config.epochs,
							verbose=config.verbose,
							validation_data=(x_val, y_val),
							callbacks=[checkpointer])
		model.load_weights(checkpoint_filepath)
	else:
			model.load_weights(checkpoint_filepath).expect_partial()

	test_score = mean_squared_error(model.predict(x_test), y_test, squared=False)
	test_scores.append(test_score)
	
	if config.train:
		val_score, best_epoch = np.sqrt(np.min(history.history['val_loss'])), np.argmin(history.history['val_loss'])+1
		train_score = np.sqrt(history.history['loss'][best_epoch-1])
		print("Best epoch {}, Train score {}, Val score {}".format(best_epoch, train_score, val_score))
		train_scores.append(train_score)
		val_scores.append(val_score)
		best_epochs.append(best_epoch)
	fold+=1

	print("Results :")
	if config.train :
		print("best_epochs {}".format(best_epochs))
		print("train_scores {:.3f}+/-{:.3f} : {}".format(np.mean(train_scores), np.std(train_scores), np.round(train_scores, 3)))
		print("val_scores {:.3f}+/-{:.3f} : {}".format(np.mean(val_scores), np.std(val_scores), np.round(val_scores, 3)))
	print("test_scores {:.3f}+/-{:.3f} : {}".format(np.mean(test_scores), np.std(test_scores), np.round(test_scores, 3)))

	if config.train :
		print("\n{:.3f}+/-{:.3f}	{:.3f}+/-{:.3f}	{:.3f}+/-{:.3f}	{}	{}	{}\n".format(np.mean(train_scores), np.std(train_scores), 
																					np.mean(val_scores), np.std(val_scores),
																					np.mean(test_scores), np.std(test_scores),
																					np.round(train_scores, 3),
																					np.round(val_scores, 3),
																					np.round(test_scores, 3),
																					))
