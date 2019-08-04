#!/usr/bin/env python3

import argparse
from enum import Enum
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../fastai/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../textblob/'))

from src import inference as fastai_inference
from src import textblob_inference


MODEL_ARGUMENT_CHOICES = {
	"fastai": "{'model_dir': Directory of the saved fastai model, 'model_name': File name of the saved fastai model}",
	"textblob": "no arguments needed"
}

class ModelType(Enum):
	'''Enum to define the different type of sentiment analysis models supported in the project.
	'''
	fastai = 0
	textblob = 1

	def __str__(self):
		return self.name

	def __repr__(self):
		return str(self)

class Model(object):
	'''Represents the sentiment analysis model.
	'''
	def __init__(self, model_type, model_args):
		'''Initializes a Model object.

		Initializes a logger.
		Defines the email labels for each polarity value.
		Initializes a sentiment analysis model.

		Args:
			model_type: A ModelType value to specify the type of sentiment analysis model to initialize.
			model_args: Arguments for the respective sentiment analysis model.
		'''
		super(Model, self).__init__()

		self.POLARITY_LABELS = { 'positive': r'positive 🤓', 'neutral': r'neutral 😶', 'negative': r'negative 🧐' }
		self.model_type = model_type
		if self.model_type is None:
			raise ValueError('no sentiment analysis model specified')

		self.initialize_models = {
			ModelType.fastai: self.initialize_fastai,
			ModelType.textblob: self.initialize_textblob
		}
		self.model = self.initialize_models[model_type](model_args)

	def initialize_fastai(self, args):
		'''Initializes the fastai classification model.

		Args:
			args: A dictionary: {'model_dir': Directory of the saved fastai model, 'model_name': File name of the saved fastai model}
		'''
		fastai_inference.initialize_model(args['model_dir'], args['model_name'])
		return fastai_inference

	def initialize_textblob(self, args):
		'''Initializes the textblob classification model.
		'''
		textblob_inference.initialize_model()
		return textblob_inference

	def analyze(self, texts_weights):
		'''Returns the polarity label after classifying texts.

		Predicts the polarity of each text.
		Determines the overall polarity through weighting.

		Args:
			texts_weights: A List of tuples: [(text, weight of text), ...]
		'''
		score = 0.0
		for text, weight in texts_weights:
			if len(text) < 1:
				continue

			text_polarity = self.model.predict(text)
			if text_polarity == 'positive':
				score = 1.0 * weight + score
			elif text_polarity == 'neutral':
				score = 0.0 * weight + score
			elif text_polarity == 'negative':
				score = -1.0 * weight + score

		total_polarity = None
		if score >= 0.5:
			total_polarity = 'positive'
		elif score > -0.5 and score < 0.5:
			total_polarity = 'neutral'
		elif score <= -0.5:
			total_polarity = 'negative'

		return self.POLARITY_LABELS[total_polarity]
