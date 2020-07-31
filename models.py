import math
import numpy as np
from tensorflow.keras.layers import Dense, Concatenate, Average, Input
import tensorflow as tf
import tensorflow.keras.backend as K

def create_feature_extractor_block(x, units):
	x = Dense(units, activation='relu')(x)
	return x

def create_mu_block(x_list, config):
	if config.n_feature_sets>1:
		if config.last_layer=='concatenate':
			x = Concatenate()(x_list)
		else:
			# pad first if prorated
			max_len = np.max(config.feature_units)
			print("Max len {}".format(max_len))
			for i, a in enumerate(x_list):
				print(i, a)
				if (max_len-config.feature_units[i])>0:
					x_list[i] = tf.pad(a, [[0, 0], [max_len-config.feature_units[i], 0]])
			x = Average()(x_list)
		if config.task=='regression':
			x = Dense(1)(x)
		elif config.task=='classification':
			x = Dense(config.n_classes, activation='softmax')(x)
	else:
		if config.task=='regression':
			x = Dense(1)(x_list[0])
		elif config.task=='classification':
			x = Dense(config.n_classes, activation='softmax')(x_list[0])
	return x

def build_model(config):

	if config.task=='classification':
		loss = 'categorical_crossentropy'
	elif config.task=='regression':
		loss = 'mse'

	n_feature_sets = len(config.feature_split_lengths)

	inputs = []
	for i in range(n_feature_sets):
		inputs.append(Input((config.feature_split_lengths[i],)))

	feature_extractors = []
	config.feature_units = []
	for i in range(n_feature_sets):
		units = config.units
		if config.build_model=='prorated':
			# print(config.feature_split_lengths[i] * config.units / sum(config.feature_split_lengths))
			units = math.floor(config.feature_split_lengths[i] * config.units / sum(config.feature_split_lengths) )
		config.feature_units.append(units)
		feature_extractors.append(create_feature_extractor_block(inputs[i], units = units))

	output = create_mu_block(feature_extractors, config)
	model = tf.keras.models.Model(inputs=inputs, outputs=output)
	if config.verbose:
		model.summary()
		tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True,
								  rankdir='TB', expand_nested=False, dpi=96)
	return model, loss
