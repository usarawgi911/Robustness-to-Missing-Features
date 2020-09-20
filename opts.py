import utils

import argparse
import os

import numpy as np
import tensorflow as tf

class Opts:
	def __init__(self):
		self.parser = argparse.ArgumentParser()

		self.subparsers = self.parser.add_subparsers(help='train | evaluate | experiment', dest='task')

		# Train Task
		self.parser_train = self.subparsers.add_parser('train', help='Train the model')

		self.parser_train.add_argument('--dataset_dir', required=True, help='Path to dataset')
		self.parser_train.add_argument('--dataset', required=True, help='Which dataset to use')
		self.parser_train.add_argument('--model_dir', default='models', help='Path to save')

		self.parser_train.add_argument('--n_folds', default=5, type=int, help='n folds to cross-validate')
		self.parser_train.add_argument('--lr', default=1e-2, type=float, help='learning rate')
		self.parser_train.add_argument('--epochs', default=1000, type=int, help='epochs')
		self.parser_train.add_argument('--batch_size', default=100, type=int, help='batch size')
		self.parser_train.add_argument('--units', default=50, type=int, help='Number of hidden units')
		self.parser_train.add_argument('--impute', default='mean', help='Missing value imputation')

		self.parser_train.add_argument('--build_model', default='prorated', help='Split units proportionately')
		self.parser_train.add_argument('--mod_split', default='computation_split', help='computation_split | none')
		self.parser_train.add_argument('--hc_threshold', default=0.5, type=float, help='Threshold for HC Clustering')
		self.parser_train.add_argument('--last_layer', default='concatenate', help='concatenate | add')

		self.parser_train.add_argument('--verbose', type=int, default=0)


		# Evaluate Task
		self.parser_evaluate = self.subparsers.add_parser('evaluate', help='Evaluate the model')

		self.parser_evaluate.add_argument('--dataset_dir', required=True, help='Path to dataset')
		self.parser_evaluate.add_argument('--dataset', required=True, help='Which dataset to use')
		self.parser_evaluate.add_argument('--model_dir', default='models', help='Path to save')

		self.parser_evaluate.add_argument('--n_folds', default=5, type=int, help='n folds to cross-validate')
		self.parser_evaluate.add_argument('--build_model', default='prorated', help='Split units proportionately')
		self.parser_evaluate.add_argument('--mod_split', default='computation_split', help='computation_split | none')
		self.parser_evaluate.add_argument('--hc_threshold', default=0.5, type=float, help='Threshold for HC Clustering')
		self.parser_evaluate.add_argument('--last_layer', default='concatenate', help='concatenate | add')
		self.parser_evaluate.add_argument('--units', default=50, type=int, help='Number of hidden units')
		self.parser_evaluate.add_argument('--impute', default='mean', help='Missing value imputation')

		self.parser_evaluate.add_argument('--verbose', type=int, default=0)

	def parse(self):
		config = self.parser.parse_args()

		return config
