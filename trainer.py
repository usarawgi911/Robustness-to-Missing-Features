import os
import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score

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
	model.load_weights(checkpoint_filepath)
	val_loss = np.sqrt(min(history.history['val_loss']))
	print("Val loss {}".format(val_loss))
	return

def train_expt2(X, y, X_test, y_test, config):
	
	train_scores, val_scores, test_scores, best_epochs = [], [], [], []
	reports = []
	for fold in range(config.n_folds):
		print("Fold {}".format(fold))
		model, loss = models.build_model(config)
		indices = [i for i in range(len(y))]
		
		if config.task=='classification':
			train_index, val_index, y_train, y_val = train_test_split(indices, y, test_size=0.2, random_state=42, stratify=y)
		elif config.task=='regression':
			train_index, val_index, y_train, y_val = train_test_split(indices, y, test_size=0.2, random_state=42)

		# if config.task=='classification':
			# print("train {} 0s, {} 1s".format(len(y_train)-np.sum(y_train), np.sum(y_train)))
			# print("val {} 0s, {} 1s".format(len(y_val)-np.sum(y_val), np.sum(y_val)))
			# print("test {} 0s, {} 1s".format(len(y_test)-np.sum(y_test), np.sum(y_test)))
		
		X_train = [X[i][train_index] for i in range(config.n_feature_sets)]
		X_val = [X[i][val_index] for i in range(config.n_feature_sets)]

		_, X_test = utils.drop_features(X, X_test, config)

		checkpoint_filepath = os.path.join(config.model_dir, config.dataset, config.expt_name, 'model_'+str(fold))
		checkpointer = tf.keras.callbacks.ModelCheckpoint(
				checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True,
				save_weights_only=True, mode='auto', save_freq='epoch')

		if config.task=='classification':
			model.compile(optimizer=tf.optimizers.Adam(learning_rate=config.lr),
						  loss=loss, metrics=['accuracy'])	
		elif config.task=='regression':
			model.compile(optimizer=tf.optimizers.Adam(learning_rate=config.lr),
						  loss=loss)
		

		y_train_ohe = np.asarray([[0, 1] if yy==1 else [1, 0] for yy in y_train])
		y_val_ohe = np.asarray([[0, 1] if yy==1 else [1, 0] for yy in y_val])
		y_test_ohe = np.asarray([[0, 1] if yy==1 else [1, 0] for yy in y_test])

		# [print('Shape of feature train set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(X_train)]
		# [print('Shape of feature val set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(X_val)]
		# [print('Shape of feature test set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(X_test)]
		# return

		history = model.fit(X_train, y_train_ohe,
							batch_size=config.batch_size,
							epochs=config.epochs,
							verbose=config.verbose,
							callbacks=[checkpointer],
							validation_data=(X_val, y_val_ohe))
		# print("Trying to load {}".format(checkpoint_filepath))
		model.load_weights(checkpoint_filepath)
		
		if config.task=='regression':
			val_score, best_epoch = np.sqrt(np.min(history.history['val_loss'])), np.argmin(history.history['val_loss'])+1
			train_score = np.sqrt(history.history['loss'][best_epoch-1])
			test_score = np.sqrt(model.evaluate(X_test, y_test, batch_size=config.batch_size, verbose=0))
		elif config.task=='classification':
			best_epoch = np.argmin(history.history['val_loss'])+1
			# best_epoch = np.argmax(history.history['val_accuracy'])+1
			train_score = history.history['accuracy'][best_epoch-1]
			val_score = history.history['val_accuracy'][best_epoch-1]
			y_preds = np.argmax(model.predict(X_test), axis=1)
			test_score = accuracy_score(y_test, y_preds)
			reports.append(classification_report(y_test, y_preds))
			print(classification_report(y_test, y_preds))
		
		print("Best epoch {}, Train score {}, Val score {}, Test score {} ".format(best_epoch, train_score, val_score, test_score))
		train_scores.append(train_score)
		val_scores.append(val_score)
		test_scores.append(test_score)
		best_epochs.append(best_epoch)
	print("Results :")
	print("best_epochs {}".format(best_epochs))
	print("train_scores {:.3f}+/-{:.3f} : {}".format(np.mean(train_scores), np.std(train_scores), np.round(train_scores, 3)))
	print("val_scores {:.3f}+/-{:.3f} : {}".format(np.mean(val_scores), np.std(val_scores), np.round(val_scores, 3)))
	print("test_scores {:.3f}+/-{:.3f} : {}".format(np.mean(test_scores), np.std(test_scores), np.round(test_scores, 3)))

	print("\n{:.3f}+/-{:.3f}	{:.3f}+/-{:.3f}	{:.3f}+/-{:.3f}	{}	{}	{}\n".format(np.mean(train_scores), np.std(train_scores), 
																					np.mean(val_scores), np.std(val_scores),
																					np.mean(test_scores), np.std(test_scores),
																					np.round(train_scores, 3),
																					np.round(val_scores, 3),
																					np.round(test_scores, 3),
																					))
	if config.task=='classification':
		np.save(os.path.join(config.model_dir, config.dataset, config.expt_name, 'reports'), np.asarray(reports))


def train_expt3(X, y, config):
	config.n_folds = 20
	kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)
	fold=1
	train_scores = []
	val_scores = []
	n_feature_sets = len(X)
	for train_index, test_index in kf.split(y):
		print('Fold {}'.format(fold))	

		y_train, y_val = y[train_index], y[test_index]
		x_train = [i[train_index] for i in X]
		x_val = [i[test_index] for i in X]

		# for i in range(n_feature_sets):
		# 		x_train[i], x_val[i] = utils.standard_scale(x_train[i], x_val[i])

		checkpoint_filepath = os.path.join(config.model_dir, config.dataset, config.expt_name, 'model_'+str(fold))
		checkpointer = tf.keras.callbacks.ModelCheckpoint(
				checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True,
				save_weights_only=True, mode='auto', save_freq='epoch')

		model, loss = models.build_model(config)
		
		model.compile(optimizer=tf.optimizers.Adam(learning_rate=config.lr),
						  loss=loss)
		history = model.fit(x_train, y_train,
							batch_size=config.batch_size,
							epochs=config.epochs,
							verbose=config.verbose,
							callbacks=[checkpointer],
							validation_data=(x_val, y_val))
		model.load_weights(checkpoint_filepath)
		
		val_score, best_epoch = np.sqrt(np.min(history.history['val_loss'])), np.argmin(history.history['val_loss'])+1
		train_score = np.sqrt(history.history['loss'][best_epoch-1])
		
		train_scores.append(train_score)
		val_scores.append(val_score)

		print("Fold {}, Best epoch {}, Train score {:.3f}, Val score {:.3f}".format(fold, best_epoch, train_score, val_score))
		
		fold += 1

	print("\n{:.3f}+/-{:.3f}	{:.3f}+/-{:.3f}	{}	{}\n".format(np.mean(train_scores), np.std(train_scores), 
																 np.mean(val_scores), np.std(val_scores),
																 np.round(train_scores, 3),
																 np.round(val_scores, 3),
																))
	