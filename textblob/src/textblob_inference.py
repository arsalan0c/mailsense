#!/usr/bin/env python3

from textblob import TextBlob
from textblob.en.sentiments import NaiveBayesAnalyzer
import nltk

def initialize_model():
	'''Initializes a textblob naive bayes sentiment analysis model.
	'''
	# prevent exposing the learner
	nltk.download('movie_reviews')
	nltk.download('punkt')
	global analyzer
	analyzer = NaiveBayesAnalyzer()
	analyzer.train()

def predict(text):
	'''Returns the polarity value after performing inference on a text.

	Uses a trained textblob sentiment analysis model.

	Args:
		text: Text to perform inference on.
	'''
	preds = analyzer.analyze(text)
	polarity = 'positive' if str(preds[0]) == 'pos' else 'negative'
	return polarity
