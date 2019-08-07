#!/usr/bin/env python3

import argparse
import pandas as pd
from fastai import *
from fastai.text import *

def load_data(path):
	df = pd.read_csv(path, index_col=[0])
	print(df.head)
	train_df = df[df[DATASET_COMPONENT_LABEL] == 'train']
	val_df = df[df[DATASET_COMPONENT_LABEL] == 'val']
	test_df = df[df[DATASET_COMPONENT_LABEL] == 'test']

	# Language model data
	data_lm = TextLMDataBunch.from_df(train_df=train_df, valid_df=val_df, test_df=test_df, text_cols='text', label_cols='polarity', path="")
	# Classifier model data
	data_clas = TextClasDataBunch.from_df(train_df=train_df, valid_df=val_df, test_df=test_df, vocab=data_lm.train_ds.vocab, bs=32, text_cols='text', label_cols='polarity', path="")
	return data_lm, data_clas

def train(data_lm, data_clas):
	learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=args.dropmult, pretrained=True)

	learn.fit_one_cycle(args.lmepochs, args.lmlearningrate)

	learn.save_encoder(LANGUAGE_MODEL_NAME)
	learn = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=args.dropmult, pretrained=True)
	learn.load_encoder(LANGUAGE_MODEL_NAME)

	learn.fit_one_cycle(args.tcepochs, args.tclearningrate)

	learn.freeze_to(-2)
	learn.fit_one_cycle(3, slice(5e-3/2., 5e-3))

	learn.freeze_to(-3)
	learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

	learn.unfreeze()
	learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))

	return learn

def show_results(learn):
	preds, y, losses = learn.get_preds(DatasetType.Train, with_loss=True)
	interp = ClassificationInterpretation(learn, preds, y, losses)
	print("\nTraining Most Confused, (Actual, Predicted, Occurrences)")
	print(interp.most_confused(slice_size=10))

	preds, y, losses = learn.get_preds(DatasetType.Valid, with_loss=True)
	interp = ClassificationInterpretation(learn, preds, y, losses)
	print("\nValidation Most Confused, (Actual, Predicted, Occurrences)")
	print(interp.most_confused(slice_size=10))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-dp', '--datasetpath', help='str: path to csv dataset', type=str, action='store', required=True)
	parser.add_argument('-od', '--outputdirectory', help='str: directory to output text classifier model file', type=str, action='store', required=True)
	parser.add_argument('-dm', '--dropmult', help='float: amount to scale dropout values by', type=float, action='store', default=0.7)
	parser.add_argument('-lmlr', '--lmlearningrate', help='float: learning rate for language model', type=float, action='store', default=1e-2)
	parser.add_argument('-lme', '--lmepochs', help='int: number of epochs to train language model', type=int, action='store', default=1)
	parser.add_argument('-tclr', '--tclearningrate', help='float: learning rate for text classifier', type=float, action='store', default=1e-2)
	parser.add_argument('-tce', '--tcepochs', help='int: number of epochs to train text classifier', type=int, action='store', default=1)
	args = parser.parse_args()

	DATASET_COMPONENT_LABEL = 'set'
	LANGUAGE_MODEL_NAME = 'languagemodel_encoder'
	TEXT_MODEL_NAME = 'textclassifier.pkl'

	data_lm, data_clas = load_data(args.datasetpath)
	learn = train(data_lm, data_clas)
	learn.export(args.outputdirectory + '/' + TEXT_MODEL_NAME)
	show_results(learn)
