#!/usr/bin/env python3

import argparse
import pandas as pd
from fastai import *
from fastai.text import *

def load_data(path):
	df = pd.read_csv(path)
	train_df = df[df[DATASET_COMPONENT_LABEL] == 'train']
	val_df = df[df[DATASET_COMPONENT_LABEL] == 'val']
	test_df = df[df[DATASET_COMPONENT_LABEL] == 'test']

	# Language model data
	data_lm = TextLMDataBunch.from_df(train_df=train_df, valid_df=val_df, test_df=test_df, text_cols=args.textcolumn, label_cols=args.labelcolumn, path="")
	# Classifier model data
	data_clas = TextClasDataBunch.from_df(train_df=train_df, valid_df=val_df, test_df=test_df, vocab=data_lm.train_ds.vocab, bs=32, text_cols=args.textcolumn, label_cols=args.labelcolumn, path="")
	return data_lm, data_clas

def train(data_lm, data_clas, epochs=1, drop_mult=1, lr=1e-2):
	# create language model with pretrained weights
	learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=drop_mult)
	# train language model using one cycle policy
	learn.fit_one_cycle(epochs, lr)
	learn.save_encoder(LANGUAGE_MODEL_NAME)
	learn = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=drop_mult)
	learn.load_encoder(LANGUAGE_MODEL_NAME)
	learn.fit_one_cycle(epochs, lr)
	return learn

def show_results(learn):
	preds, y, losses = learn.get_preds(DatasetType.Train, with_loss=True)
	interp = ClassificationInterpretation(learn, preds, y, losses)
	print(interp.confusion_matrix(slice_size=10))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-dp', '--datasetpath', help='str: path to csv dataset', type=str, action='store', required=True)
	parser.add_argument('-tcol', '--textcolumn', help='int: column index of text', type=int, action='store', required=True)
	parser.add_argument('-lcol', '--labelcolumn', help='int: column index of labels', type=int, action='store', required=True)
	parser.add_argument('-dm', '--dropmult', help='float: amount to scale dropout values by', type=float, action='store', default=1)
	parser.add_argument('-lr', '--learningrate', help='float: learning rate for both language and text models', type=float, action='store', default=1e-2)
	parser.add_argument('-e', '--epochs', help='int: number of epochs to train both language and text models', type=int, action='store', default=1)
	args = parser.parse_args()

	DATASET_COMPONENT_LABEL = 'set'
	LANGUAGE_MODEL_NAME = 'languagemodel_encoder'
	TEXT_MODEL_NAME = 'textclassifier.pkl'

	data_lm, data_clas = load_data(args.datasetpath)
	learn = train(data_lm, data_clas, args.epochs, args.dropmult, args.learningrate)
	learn.export('models/' + TEXT_MODEL_NAME)
	show_results(learn)