#!/usr/bin/env python3

import argparse
import enum

class ModelType(enum.Enum):
	'''
	'''
	fastai = 1


class Model(object):
	'''
	'''
	def __init__(self, model_type=None, model_args=None):
		super(Model, self).__init__()
		self.model_type = model_type
		self.initialize_models = {
			None: self.no_model_provided,
			ModelType.fastai: self.initialize_fastai
		}

		self.initialize_models.get(model_type, self.no_model_provided)()

	def no_model_provided(self):
		'''
		'''
		err = True

	def initialize_fastai(self, args):
		'''
		'''




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='process arguments required for subscriber and mail functionality')
	parser.add_argument('-p', '--project', help='string: name of the project from Google Cloud', type=str, action='store', required=True)
