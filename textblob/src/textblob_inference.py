#!/usr/bin/env python3

from textblob import TextBlob
from textblob.en.sentiments import NaiveBayesAnalyzer
import nltk

def initialize_model():
	'''Initializes a textblob naive bayes sentiment analysis model.
	'''
	nltk.download('movie_reviews')
	nltk.download('punkt')

	# prevent exposing the learner
	global analyzer
	analyzer = NaiveBayesAnalyzer()
	analyzer.train()

def predict(text):
	'''Returns the polarity value after performing inference on a text.

	Uses a trained textblob sentiment analysis model.
	If the difference between positive and negative probabilities is too small, the result is considered neutral.

	Args:
		text: Text to perform inference on.
	'''
	MIN_DELTA = 0.1

	preds = analyzer.analyze(text)
	p_pos = preds[1]
	p_neg = preds[2]
	diff = p_pos - p_neg
	if diff >= MIN_DELTA:
		return 'positive'
	elif diff <= -MIN_DELTA:
		return 'negative'
	else:
		return 'neutral'
