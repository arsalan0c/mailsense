#!/usr/bin/env python3

# This script is used to perform inference to classify a text using a trained Fast.ai language model.

from fastai import *
from fastai.text import *

import argparse

def initialize_model(model_dir, model_name):
	'''Initializes a saved Fast.ai language model.

	Args:
		model_dir: Directory path for a Fast.ai language model.
		model_name: File name for a Fast.ai language model.
	'''
	# prevent exposing the learner
	global learn
	learn = load_learner(model_dir, model_name)

def predict(text):
	'''Returns the polarity label after performing inference on a text.

	Uses a trained Fast.ai language model.

	Args:
		text: Text to perform inference on.
	'''
	preds = learn.predict(text)
	polarity = str(preds[0])
	return polarity

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='process arguments required for performing inference with trained model on input text')
	parser.add_argument('-md', '--modeldirectory', help='string: path to directory containing model file', type=str, action='store', required=True)
	parser.add_argument('-mn', '--modelname', help='string: name of model file', type=str, action='store', required=True)
	parser.add_argument('-t', '--text', help='string: text to perform inference on', type=str, action='store', required=True)
	args = parser.parse_args()

	initialize_model(args.modeldirectory, args.modelname)
	polarity = predict(args.text)
	print(polarity)
