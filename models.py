import math
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
			x = Average()(x_list)
		x = Dense(1)(x)
	else:
		x = Dense(1)(x_list[0])
	return x

def build_model(config):

	loss = 'mse'
	n_feature_sets = len(config.feature_split_lengths)

	inputs = []
	for i in range(n_feature_sets):
		inputs.append(Input((config.feature_split_lengths[i],)))

	feature_extractors = []
	for i in range(n_feature_sets):
		units = config.units
		if config.build_model=='prorated':
			# print(config.feature_split_lengths[i] * config.units / sum(config.feature_split_lengths))
			units = math.floor(config.feature_split_lengths[i] * config.units / sum(config.feature_split_lengths) )
		feature_extractors.append(create_feature_extractor_block(inputs[i], units = units))

	output = create_mu_block(feature_extractors, config)
	model = tf.keras.models.Model(inputs=inputs, outputs=output)
	if config.verbose:
		model.summary()
		tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True,
								  rankdir='TB', expand_nested=False, dpi=96)
	return model, loss