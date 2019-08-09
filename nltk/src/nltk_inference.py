#!/usr/bin/env python3

import argparse

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

def initialize_model():
	'''Initializes a nltk sentiment intensity analysis model.
	'''
	# prevent exposing the learner
	global analyzer
	analyzer = SIA()

def predict(text):
	'''Returns the polarity value after performing inference on a text.

	Uses a nltk sentiment analysis model.

	Args:
		text: Text to perform inference on.
	'''
	preds = analyzer.polarity_scores(text)
	if 'compound' in preds:
		del preds['compound']

	# get the key with the max value
	polarity = max(preds, key=preds.get)
	if polarity == 'pos':
		return 'positive'
	elif polarity == 'neu':
		return 'neutral'
	elif polarity == 'neg':
		return 'negative'

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='process arguments required for performing inference with trained model on input text')
	parser.add_argument('-t', '--text', help='string: text to perform inference on', type=str, action='store', required=True)
	args = parser.parse_args()

	initialize_model()
	polarity = predict(args.text)
	print(polarity)
